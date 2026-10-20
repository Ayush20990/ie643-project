"""Fast local dynamics fits and source-guarded next-hour neural/dynamics mixtures."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from .config import MODEL_DIR,PROC_DIR,RESULTS_DIR
from .data import batches,save_meta
from .diagnose_peaks import peak_metrics
from .improve import evaluate,fixed_scale,write_json
from .nowcast_hybrid import LocalDynamics,NextHourHybrid,dynamics_features
from .nowcast_tcn import within_input,turning_weights
from .peak_data import prepare_peak
from .progressive_train import sha
from .refined_models import ridge_path
from .train import predict

FAMILY="next_hour_hybrid"
RIDGES=(.00001,.0001,.001,.01,.1,1.,10.)
ANCHORS=(.00001,.0001,.001,.01,.1,1.,10.)


def sample(x,y):
    ix=np.linspace(0,len(x)-1,min(12000,len(x)),dtype=int)
    return x[ix],y[ix]


def stats(x,y,channels,key):
    gram=rhs=None
    for xb,yb in batches(x,y,bs=512,shuffle=False):
        f,last=dynamics_features(xb,channels)
        a=f.permute(1,0,2).double()
        b=(yb[:,0]-last).T[:,:,None].double()
        # Turning-weighted source/SSL loss for key; others use ordinary MSE.
        w=torch.ones((channels,len(xb),1),dtype=torch.float64)
        w[key,:,0]=.5+.5*turning_weights(xb,yb,key).double()
        g=a.transpose(1,2)@(a*w);r=a.transpose(1,2)@(b*w)
        gram,rhs=(g,r) if gram is None else (gram+g,rhs+r)
    return gram/len(x),rhs/len(x)


@torch.no_grad()
def errors(model,x,y,key):
    p=predict(model,x,bs=256)[:,0]
    e=(p-y[:,0].numpy())**2
    loss=e.copy();w=turning_weights(x,y,key).numpy()
    # Half ordinary, half reversal-weighted MSE, only on the key channel.
    loss[:,key]*=.5+.5*w/np.mean(w)
    return e.mean(0),loss.mean(0)


def fit_local(ds,args):
    data=prepare_peak(ds);cfg={**data["cfg"],"H":1}
    src,tgt=data["source"],data["target"]
    out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    out.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    if (out/"local_selection.json").exists():return data
    c,key=len(cfg["columns"]),cfg["columns"].index(cfg["key"])
    st=sample(src["train_packed"][0],src["train_packed"][1][:,:1])
    sv=(src["val_packed"][0][::25],src["val_packed"][1][::25,:1])
    tx=tgt["train_packed"][0]
    ix=np.linspace(0,len(tx)-1,min(12000,len(tx)),dtype=int)
    tt=within_input(tx[ix],c)
    tv=within_input(tgt["val_packed"][0][cfg["L"]::25],c)
    gs,rs=stats(*st,c,key);gt,rt=stats(*tt,c,key)
    source=LocalDynamics(cfg)
    mse,score=errors(source,*sv,key)
    best=score.copy();weight=source.weight.clone()
    source_choices=[{"channel":j,"ridge":None,"MSE":float(mse[j]),"score":float(score[j])} for j in range(c)]
    source_rows=[]
    for ridge,w in ridge_path(gs,rs,RIDGES):
        source.weight.copy_(w);m,s=errors(source,*sv,key)
        for j in range(c):
            row={"channel":j,"ridge":ridge,"MSE":float(m[j]),"score":float(s[j]),"eligible":bool(m[j]<=1.05*mse[j])}
            source_rows.append(row)
            if row["eligible"] and s[j]<best[j]:
                weight[j]=w[j];best[j]=s[j];source_choices[j]=row
    source.weight.copy_(weight)
    adapter=copy.deepcopy(source)
    bs,qs=errors(source,*sv,key);bt,qt=errors(source,*tv,key)
    best=np.ones(c);chosen=weight.clone()
    choices=[{"channel":j,"recipe":"source_fallback","proxy":1.} for j in range(c)]
    rows=[]
    for anchor,w in ridge_path(gt+.5*gs,rt+.5*rs,ANCHORS,source.weight):
        adapter.weight.copy_(w)
        ms,es=errors(adapter,*sv,key);mt,et=errors(adapter,*tv,key)
        proxy=.5*es/qs.clip(1e-8)+.5*et/qt.clip(1e-8)
        for j in range(c):
            eligible=bool(ms[j]<=1.1*bs[j])
            row={"channel":j,"anchor":anchor,"source_MSE":float(ms[j]),"target_input_MSE":float(mt[j]),"proxy":float(proxy[j]),"eligible":eligible}
            rows.append(row)
            if eligible and proxy[j]<best[j]:best[j]=proxy[j];chosen[j]=w[j];choices[j]=row
    adapter.weight.copy_(chosen)
    torch.save(source.state_dict(),dest/"local_source.pt");torch.save(adapter.state_dict(),dest/"local_adapted.pt")
    write_json(out/"local_selection.json",{"source":source_choices,"adapted":choices,"target_future_labels_used":False,
        "objective":"raw-unit next-hour reversal-weighted ridge; target inner-X loss + .5 source replay + source-weight anchor"})
    pd.DataFrame(source_rows).to_csv(out/"local_source_search.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"local_adaptation_search.csv",index=False)
    return data


@torch.no_grad()
def components(model,x):return torch.cat([model.components(xb) for xb in batches(x,bs=256,shuffle=False)]).numpy()


def fit_mixture(model,source_val,target_val):
    x,y=source_val;xt,yt=target_val
    ps,pt=components(model,x)[:,0],components(model,xt)[:,0]
    ys,yt=y[:,0].numpy(),yt[:,0].numpy()
    key=model.C-1  # replaced explicitly below by caller's model key
    key=model.key_index
    ws,wt=np.ones_like(ys),np.ones_like(yt)
    a=turning_weights(x,y,key).numpy();b=turning_weights(xt,target_val[1],key).numpy()
    ws[:,key]=.5+.5*a/a.mean();wt[:,key]=.5+.5*b/b.mean()
    source_mix=np.zeros((model.C,5));adapt_mix=np.zeros_like(source_mix);records=[]
    for j in range(model.C):
        s,t=ps[:,j].astype(float),pt[:,j].astype(float)
        yy,tt=ys[:,j],yt[:,j]
        gs=s.T@(s*ws[:,j,None])/len(s);rs=s.T@(yy*ws[:,j])/len(s)
        gt=t.T@(t*wt[:,j,None])/len(t);rt=t.T@(tt*wt[:,j])/len(t)
        g0=s.T@s/len(s);r0=s.T@yy/len(s);ss=np.mean(yy.astype(float)**2)
        source_error=lambda w:float(w@g0@w-2*w@r0+ss)
        # Source simplex permits source TCN, source dynamics and persistence only.
        initial=np.array([0.,0.,0.,0.,1.])
        fit=minimize(lambda w:float(w@gs@w-2*w@rs),initial,jac=lambda w:2*(gs@w-rs),method="SLSQP",
            bounds=[(0,1),(0,0),(0,1),(0,0),(0,1)],constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
            options={"ftol":1e-12,"maxiter":300})
        w=np.maximum(fit.x,0);w/=w.sum()
        if not fit.success or source_error(w)>1.05*source_error(initial):w=initial
        source_mix[j]=w
        bs=max(float(np.mean((s@w-yy)**2*ws[:,j])),1e-8)
        bt=max(float(np.mean((t@w-tt)**2*wt[:,j])),1e-8)
        source_ref=source_error(w)
        start=w.copy()
        objective=lambda v:.5*np.mean((s@v-yy)**2*ws[:,j])/bs+.5*np.mean((t@v-tt)**2*wt[:,j])/bt+.001*np.sum((v-start)**2)
        jac=lambda v:(gs@v-rs)/bs+(gt@v-rt)/bt+.002*(v-start)
        adapted=minimize(objective,start,jac=jac,method="SLSQP",bounds=[(0,1)]*5,
            constraints=[{"type":"eq","fun":lambda v:v.sum()-1},
                         {"type":"ineq","fun":lambda v:1.1*source_ref-source_error(v)}],
            options={"ftol":1e-11,"maxiter":400})
        v=np.maximum(adapted.x,0);v/=v.sum()
        accepted=bool(adapted.success and source_error(v)<=1.1*source_ref+1e-8 and objective(v)<=objective(start)+1e-9)
        if not accepted:v=start
        adapt_mix[j]=v
        records.append({"channel":j,"source_weights":w.tolist(),"adapted_weights":v.tolist(),"proxy":float(objective(v)),"accepted":accepted,
            "source_reference_MSE":source_ref,"adapted_source_MSE":source_error(v),"solver":str(adapted.message)})
    return source_mix,adapt_mix,records


def fit_dataset(ds,args):
    data=fit_local(ds,args);cfg={**data["cfg"],"H":1}
    src,tgt=data["source"],data["target"]
    out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    manifest={"dataset":ds,"seeds":args.seeds,"local_ridges":RIDGES,"local_anchors":ANCHORS,"local_replay":.5,
        "target_future_labels_used_for_training_or_selection":False,"evaluation":"rolling next-hour with hourly updated observations",
        "test_status":"reused development; no independent confirmation",
        "code_sha256":{n:sha(Path(__file__).with_name(n)) for n in ("nowcast_hybrid.py","nowcast_hybrid_train.py","nowcast_tcn.py","nowcast_train.py","peak_data.py","data.py")},
        "data_sha256":{p.name:sha(p) for p in sorted((PROC_DIR/ds).glob("*.csv"))},
        "base_sha256":{str(p.relative_to(MODEL_DIR)):sha(p) for s in args.seeds for p in (MODEL_DIR/"next_hour"/ds/f"seed_{s}"/"source_model.pt",MODEL_DIR/"next_hour"/ds/f"seed_{s}"/"adapted_model.pt")}}
    path=out/"manifest.json"
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(manifest)):raise ValueError("Run changed")
    write_json(path,manifest)
    sv=(src["val_packed"][0][::25],src["val_packed"][1][::25,:1])
    tv=within_input(tgt["val_packed"][0][cfg["L"]::25],len(cfg["columns"]))
    for seed in args.seeds:
        model=NextHourHybrid(cfg);model.key_index=cfg["columns"].index(cfg["key"])
        for role in ("source","adapted"):
            model.models[f"tcn_{role}"].load_state_dict(torch.load(MODEL_DIR/"next_hour"/ds/f"seed_{seed}"/f"{role}_model.pt",weights_only=True))
            model.models[f"dynamics_{role}"].load_state_dict(torch.load(dest/f"local_{role}.pt",weights_only=True))
        model.eval();source_mix,adapted_mix,records=fit_mixture(model,sv,tv)
        model.mix.copy_(torch.from_numpy(adapted_mix.astype(np.float32)))
        adapted=copy.deepcopy(model)
        model.mix.copy_(torch.from_numpy(source_mix.astype(np.float32)))
        model.models["tcn_adapted"].load_state_dict(model.models["tcn_source"].state_dict())
        model.models["dynamics_adapted"].load_state_dict(model.models["dynamics_source"].state_dict())
        seed_dir=dest/f"seed_{seed}";seed_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(),seed_dir/"source_model.pt");torch.save(adapted.state_dict(),seed_dir/"adapted_model.pt")
        selection={"seed":seed,"calibration":records,"target_future_labels_used_for_training_or_selection":False}
        write_json(out/f"seed_{seed}_selection.json",selection)
        if seed==0:
            save_meta(ds,{**data,"cfg":cfg},dest/"meta.json")
            meta=json.loads((dest/"meta.json").read_text());meta.update(model_spec={"architecture":FAMILY},training_seed=0,
                experiment=FAMILY,adaptation="source_anchored_local_dynamics_and_neural_mixture",forecast_semantics="next-hour rolling, observations updated every hour")
            write_json(dest/"meta.json",meta);write_json(dest/"selection.json",selection)
            for name in ("source_model","adapted_model"):(dest/f"{name}.pt").write_bytes((seed_dir/f"{name}.pt").read_bytes())
    write_json(out/"selection_frozen.json",{"timestamp_utc":pd.Timestamp.now(tz="UTC").isoformat(),"test_evaluated_at_freeze":False,
        "checkpoint_sha256":{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*.pt"))}})


def evaluate_dataset(ds,args):
    data=prepare_peak(ds);cfg={**data["cfg"],"H":1};out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    rows,peaks,examples=[],[],{}
    for seed in args.seeds:
        for name in ("source_model","adapted_model"):
            model=NextHourHybrid(cfg);model.load_state_dict(torch.load(dest/f"seed_{seed}"/f"{name}.pt",weights_only=True))
            for role in ("source","target"):
                dom=data[role];x,y=dom["test_packed"];truth=y[:,:1];p=predict(model,x,bs=256)
                scores,_=evaluate(ds,cfg,p,{**dom,"test":(dom["test"][0],truth)},fixed_scale(ds,role,dom["scaler"]))
                rows.append({"dataset":ds,"seed":seed,"domain":role,"method":name,**scores})
                key=cfg["columns"].index(cfg["key"]);sd=dom["scaler"].std[key];n=(len(p)//24)*24
                peaks.append({"dataset":ds,"seed":seed,"domain":role,"method":name,
                    "semantics":"rolling 1h, observations updated hourly",**peak_metrics(truth[:n,0,key].numpy().reshape(-1,24)*sd,p[:n,0,key].reshape(-1,24)*sd)})
                if seed==0:
                    examples[f"{role}_{name}"]=p;examples[f"{role}_truth"]=truth.numpy();examples[f"{role}_X"]=dom["test"][0].numpy()
    f=pd.DataFrame(rows);f.to_csv(out/"metrics_all.csv",index=False);pd.DataFrame(peaks).to_csv(out/"rolling_peak_metrics.csv",index=False)
    np.savez_compressed(out/"rolling_forecasts.npz",**examples)
    print(ds+" HYBRID NEXT-HOUR\n"+f.query("domain=='target'").groupby("method")[["MSE","key_RMSE"]].mean().to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--datasets",nargs="+",default=["climate","energy"])
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2]);p.add_argument("--fit-local-only",action="store_true")
    args=p.parse_args();torch.set_num_threads(2)
    if args.fit_local_only:
        for ds in args.datasets:fit_local(ds,args)
    else:
        for ds in args.datasets:fit_dataset(ds,args)
        for ds in args.datasets:evaluate_dataset(ds,args)


if __name__=="__main__":main()
