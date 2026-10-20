"""Peak/dip-aligned adaptation from rolling episodes wholly inside observed X.

Source/target experts remain frozen. Only a small causal-regime simplex is fitted.
No target original Y tensor is accepted by the calibration data collector.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from .config import MODEL_DIR,PROC_DIR,RESULTS_DIR
from .data import save_meta
from .diagnose_peaks import peak_metrics
from .improve import evaluate,fixed_scale,write_json
from .peak_data import prepare_peak
from .peak_nowcast import PeakNowcast,observed_regime,episode_prefix
from .progressive_train import sha
from .train import predict

FAMILY="peak_next_hour"
EVENT_WEIGHT=2.
RAMP_WEIGHT=.1


@torch.no_grad()
def collect_episodes(model,input_windows):
    """Prefixes/labels come from this X only; no cross-window assembly or Y."""
    predictions,regimes,truth=[],[],[]
    model.eval()
    for end in range(model.L-24,model.L):
        prefix,tail=episode_prefix(input_windows,end,model.C)
        predictions.append(model.experts(prefix))
        regimes.append(observed_regime(prefix,model.C)[:,None])
        truth.append(tail)
    return torch.cat(predictions,1).numpy(),torch.cat(regimes,1).numpy(),torch.cat(truth,1).numpy()


def design(predictions,regimes,truth,channel):
    p=predictions[:,:,channel].astype(float);reg=regimes[:,:,channel]
    y=truth[:,:,channel].astype(float);e,h=y.shape
    a=np.zeros((e,h,20))
    flat=a.reshape(-1,20)
    for r in range(4):flat[reg.reshape(-1)==r,r*5:(r+1)*5]=p.reshape(-1,5)[reg.reshape(-1)==r]
    w=np.ones_like(y);i=np.arange(e)
    w[i,y.argmax(1)]+=EVENT_WEIGHT*h/2
    w[i,y.argmin(1)]+=EVENT_WEIGHT*h/2
    da=np.diff(a,axis=1).reshape(-1,20);dy=np.diff(y,axis=1).reshape(-1)
    a=a.reshape(-1,20);yy=y.reshape(-1);ww=w.reshape(-1)
    g=a.T@(a*ww[:,None])/len(yy)+RAMP_WEIGHT*da.T@da/len(dy)
    rhs=a.T@(yy*ww)/len(yy)+RAMP_WEIGHT*da.T@dy/len(dy)
    g0=a.T@a/len(yy);r0=a.T@yy/len(yy);constant=np.mean(yy**2)
    event=lambda v:float(np.mean((a@v-yy)**2*ww)+RAMP_WEIGHT*np.mean((da@v-dy)**2))
    point=lambda v:float(v@g0@v-2*v@r0+constant)
    return g,rhs,event,point,np.bincount(reg.reshape(-1),minlength=4)


def normalize(v):
    v=np.maximum(v.reshape(4,5),0)
    return (v/v.sum(1,keepdims=True)).reshape(-1)


def fit_weights(source_data,target_data,channels):
    source_mix,adapted_mix,records=[],[],[]
    constraints=[{"type":"eq","fun":lambda v,r=r:v.reshape(4,5)[r].sum()-1} for r in range(4)]
    for channel in range(channels):
        gs,rs,ls,ms,ns=design(*source_data,channel)
        gt,rt,lt,mt,nt=design(*target_data,channel)
        # Choose a feasible source-only point baseline, then optimize event errors.
        starts=[]
        for expert in (0,2,4):
            v=np.zeros((4,5));v[:,expert]=1.;starts.append(v.reshape(-1))
        initial=min(starts,key=ms)
        bounds=[]
        for r in range(4):
            for j in range(5):
                bounds.append((initial[r*5+j],initial[r*5+j]) if ns[r]<64 else (0,0) if j in (1,3) else (0,1))
        reference=ms(initial)
        source_fit=minimize(lambda v:float(v@gs@v-2*v@rs+.001*max(reference,1e-5)*np.sum((v-initial)**2)),initial,
            jac=lambda v:2*(gs@v-rs)+.002*max(reference,1e-5)*(v-initial),bounds=bounds,
            constraints=constraints+[{"type":"ineq","fun":lambda v:1.05*reference-ms(v)}],method="SLSQP",
            options={"ftol":1e-12,"maxiter":500})
        source=normalize(source_fit.x)
        if not source_fit.success or ms(source)>1.05*reference+1e-8:source=initial
        bs,bt=max(ls(source),1e-8),max(lt(source),1e-8)
        point_s,point_t=ms(source),mt(source)
        objective=lambda v:.5*ls(v)/bs+.5*lt(v)/bt+.001*np.sum((v-source)**2)
        jac=lambda v:(gs@v-rs)/bs+(gt@v-rt)/bt+.002*(v-source)
        adapted_bounds=[]
        for r in range(4):
            for j in range(5):adapted_bounds.append((source[r*5+j],source[r*5+j]) if min(ns[r],nt[r])<64 else (0,1))
        fit=minimize(objective,source,jac=jac,bounds=adapted_bounds,method="SLSQP",
            constraints=constraints+[{"type":"ineq","fun":lambda v:1.1*point_s-ms(v)},
                                     {"type":"ineq","fun":lambda v:1.05*point_t-mt(v)}],
            options={"ftol":1e-11,"maxiter":700})
        adapted=normalize(fit.x)
        accepted=bool(fit.success and ms(adapted)<=1.1*point_s+1e-8 and mt(adapted)<=1.05*point_t+1e-8 and objective(adapted)<=objective(source)+1e-9)
        if not accepted:adapted=source
        source_mix.append(source.reshape(4,5));adapted_mix.append(adapted.reshape(4,5))
        records.append({"channel":channel,"source_regime_counts":ns.tolist(),"target_input_regime_counts":nt.tolist(),
            "source_weights":source.reshape(4,5).tolist(),"adapted_weights":adapted.reshape(4,5).tolist(),
            "source_point_reference":point_s,"adapted_source_point":ms(adapted),
            "target_input_point_reference":point_t,"adapted_target_input_point":mt(adapted),
            "source_event_reference":bs,"adapted_source_event":ls(adapted),
            "target_input_event_reference":bt,"adapted_target_input_event":lt(adapted),
            "proxy":float(objective(adapted)),"accepted":accepted,"solver":str(fit.message)})
    return np.stack(source_mix),np.stack(adapted_mix),records


def initial_model(ds,seed,cfg):
    ref_family="strict_context" if ds=="climate" else "strict_refined"
    ref_record=json.loads((RESULTS_DIR/ref_family/ds/f"seed_{seed}_selection.json").read_text())
    neural_spec=ref_record["model_spec"]["neural_spec"]
    model=PeakNowcast(cfg,"context" if ds=="climate" else "strict",neural_spec)
    current_dir=MODEL_DIR/"next_hour_hybrid"/ds/f"seed_{seed}"
    cs=torch.load(current_dir/"source_model.pt",weights_only=True);ca=torch.load(current_dir/"adapted_model.pt",weights_only=True)
    model.current.load_state_dict(ca);model.current_source_mix.copy_(cs["mix"]);model.current_adapted_mix.copy_(ca["mix"])
    ref_dir=MODEL_DIR/ref_family/ds/f"seed_{seed}"
    rs=torch.load(ref_dir/"source_model.pt",weights_only=True);ra=torch.load(ref_dir/"adapted_model.pt",weights_only=True)
    assert all(torch.equal(v,ra[n]) for n,v in rs.items() if n!="mix")
    model.reference.load_state_dict(ra);model.reference_source_mix.copy_(rs["mix"]);model.reference_adapted_mix.copy_(ra["mix"])
    return model.eval(),neural_spec


def fit_dataset(ds,args):
    data=prepare_peak(ds);cfg={**data["cfg"],"H":1}
    out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    out.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    manifest={"dataset":ds,"seeds":args.seeds,"event_weight":EVENT_WEIGHT,"ramp_weight":RAMP_WEIGHT,
        "regimes":"regular, approaching peak, approaching trough, flat/observed reversal; past last 3 values and past 48-hour scale only",
        "calibration":"rolling 24-hour episodes wholly inside each source/target validation X; first L origins purged; stride25",
        "objective":"point MSE + 2x actual peak/trough MSE + .1 ramp MSE; equal relative source/target-input event loss + .001 source-mixture anchor",
        "guards":"source point degradation <=10%, target-input point degradation <=5%, minimum64 points per regime; source fallback",
        "target_future_labels_used_for_training_or_selection":False,"cross_window_target_label_reconstruction":False,
        "test_status":"previously inspected development period; no independent confirmation",
        "code_sha256":{n:sha(Path(__file__).with_name(n)) for n in ("peak_nowcast.py","peak_nowcast_train.py","nowcast_hybrid.py","nowcast_tcn.py","context_models.py","strict_models.py","refined_models.py","model.py","data.py")},
        "data_sha256":{p.name:sha(p) for p in sorted((PROC_DIR/ds).glob("*.csv"))},
        "expert_sha256":{str(p.relative_to(MODEL_DIR)):sha(p) for s in args.seeds for fam in
            ("next_hour_hybrid","strict_context" if ds=="climate" else "strict_refined") for p in
            (MODEL_DIR/fam/ds/f"seed_{s}"/"source_model.pt",MODEL_DIR/fam/ds/f"seed_{s}"/"adapted_model.pt")}}
    path=out/"manifest.json"
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(manifest)):raise ValueError("Run changed")
    write_json(path,manifest)
    sx=data["source"]["val_packed"][0][cfg["L"]::25]
    tx=data["target"]["val_packed"][0][cfg["L"]::25]
    for seed in args.seeds:
        print(f"{ds} seed={seed}: fit observed-cycle peak/dip calibration",flush=True)
        model,spec=initial_model(ds,seed,cfg)
        source_data=collect_episodes(model,sx);target_data=collect_episodes(model,tx)
        sm,am,records=fit_weights(source_data,target_data,model.C)
        model.mix.copy_(torch.from_numpy(am.astype(np.float32)))
        source=copy.deepcopy(model);source.mix.copy_(torch.from_numpy(sm.astype(np.float32)));source.remove_inactive_target_weights()
        seed_dir=dest/f"seed_{seed}";seed_dir.mkdir(exist_ok=True)
        torch.save(source.state_dict(),seed_dir/"source_model.pt");torch.save(model.state_dict(),seed_dir/"adapted_model.pt")
        selection={"seed":seed,"calibration":records,"neural_spec":spec,"target_future_labels_used_for_training_or_selection":False}
        write_json(out/f"seed_{seed}_selection.json",selection)
        if seed==0:
            save_meta(ds,{**data,"cfg":cfg},dest/"meta.json")
            meta=json.loads((dest/"meta.json").read_text());meta.update(model_spec={"architecture":FAMILY,"reference_kind":"context" if ds=="climate" else "strict","neural_spec":spec},
                training_seed=0,experiment=FAMILY,adaptation="regime_conditioned_observed_episode_peak_loss",forecast_semantics="next-hour rolling; observations refreshed each hour")
            write_json(dest/"meta.json",meta);write_json(dest/"selection.json",selection)
            for name in ("source_model","adapted_model"):(dest/f"{name}.pt").write_bytes((seed_dir/f"{name}.pt").read_bytes())
    write_json(out/"selection_frozen.json",{"timestamp_utc":pd.Timestamp.now(tz="UTC").isoformat(),"test_evaluated_at_freeze":False,
        "checkpoint_sha256":{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*.pt"))}})


def evaluate_dataset(ds,args):
    data=prepare_peak(ds);cfg={**data["cfg"],"H":1};out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    rows,peaks,examples=[],[],{}
    for seed in args.seeds:
        spec=json.loads((out/f"seed_{seed}_selection.json").read_text())["neural_spec"]
        for name in ("source_model","adapted_model"):
            model=PeakNowcast(cfg,"context" if ds=="climate" else "strict",spec)
            model.load_state_dict(torch.load(dest/f"seed_{seed}"/f"{name}.pt",weights_only=True))
            for role in ("source","target"):
                dom=data[role];x,y=dom["test_packed"];truth=y[:,:1]
                p=predict(model,x,bs=256)
                m,_=evaluate(ds,cfg,p,{**dom,"test":(dom["test"][0],truth)},fixed_scale(ds,role,dom["scaler"]))
                rows.append({"dataset":ds,"seed":seed,"domain":role,"method":name,**m})
                key=cfg["columns"].index(cfg["key"]);sd=dom["scaler"].std[key];n=(len(p)//24)*24
                peaks.append({"dataset":ds,"seed":seed,"domain":role,"method":name,
                    **peak_metrics(truth[:n,0,key].numpy().reshape(-1,24)*sd,p[:n,0,key].reshape(-1,24)*sd)})
                if seed==0:
                    examples[f"{role}_{name}"]=p;examples[f"{role}_truth"]=truth.numpy();examples[f"{role}_X"]=dom["test"][0].numpy()
    f=pd.DataFrame(rows);f.to_csv(out/"metrics_all.csv",index=False)
    pd.DataFrame(peaks).to_csv(out/"rolling_peak_metrics.csv",index=False)
    np.savez_compressed(out/"rolling_forecasts.npz",**examples)
    print(ds+" PEAK-CALIBRATED NEXT-HOUR\n"+f.query("domain=='target'").groupby("method")[["MSE","key_RMSE"]].mean().to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--datasets",nargs="+",default=["climate","energy"])
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2]);args=p.parse_args();torch.set_num_threads(2)
    for ds in args.datasets:fit_dataset(ds,args)
    for ds in args.datasets:evaluate_dataset(ds,args)


if __name__=="__main__":main()
