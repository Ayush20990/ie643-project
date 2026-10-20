"""Temperature-only vs predicted-driver decoder on frozen source LiteTCNs.

Source validation and individual target input tails are the only selection data.
The previously inspected October follow-up is not independent for this experiment.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import MODEL_DIR, PROC_DIR, RESULTS_DIR
from .coupled_tcn import CoupledLiteTCN
from .data import batches, save_meta
from .diagnose_peaks import peak_metrics
from .improve import evaluate, fixed_scale, write_json
from .peak_data import prepare_peak
from .progressive_tcn import forecast_objective, input_pair
from .progressive_train import sha
from .train import predict, set_seed

FAMILY="coupled_litetcn"
CONTROLS=("temperature_only","coupled")


def cache(model,x,y):
    parts=[[],[],[]]
    for xb in batches(x,bs=256,shuffle=False):
        for target,value in zip(parts,model.cache_features(xb)):
            target.append(value)
    return tuple(torch.cat(v) for v in parts)+(y,)


def cache_input(model,x):
    # Construct each pretext separately; no reconstruction across windows.
    prefix,tail=input_pair(x,model.H)
    return cache(model,prefix,tail[...,:model.C])


@torch.no_grad()
def score(model,cached):
    squared=key_squared=0.
    features,base,scale,y=cached
    for ix in torch.arange(len(y)).split(256):
        p=model.from_features(features[ix],base[ix],scale[ix])
        error=(p-y[ix]).square()
        squared+=float(error.mean())*len(ix)/len(y)
        key_squared+=float(error[...,model.key].mean())*len(ix)/len(y)
    return {"MSE":squared,"key_MSE":key_squared,"score":.5*squared+.5*key_squared}


def fit(model,source_train,source_val,target_train,target_val,seed,args,adapt,log):
    set_seed(seed+int(adapt)*1000)
    gen=torch.Generator().manual_seed(seed+int(adapt)*1000)
    opt=torch.optim.AdamW(model.decoder.parameters(),lr=1e-4 if adapt else 5e-4,weight_decay=.001)
    rounds=args.adapt_rounds if adapt else args.rounds
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,rounds,eta_min=1e-6)
    ref={n:p.detach().clone() for n,p in model.decoder.named_parameters()}
    bs=score(model,source_val)
    bt=score(model,target_val) if adapt else None
    best,chosen,best_score=copy.deepcopy(model.state_dict()),0,1. if adapt else bs["score"]
    history=[{"round":0,"source":bs,"target_input":bt,"proxy":best_score,"eligible":True}]
    for r in range(1,rounds+1):
        model.train()
        primary=target_train if adapt else source_train
        ix=torch.randperm(len(primary[-1]),generator=gen)[:args.max_windows]
        si=torch.randint(len(source_train[-1]),(len(ix),),generator=gen)
        total=0.
        for i,j in zip(ix.split(256),si.split(256)):
            p=model.from_features(*(v[i] for v in primary[:3]))
            loss=forecast_objective(p,primary[-1][i],model.key)
            if adapt:
                replay=model.from_features(*(v[j] for v in source_train[:3]))
                loss=loss+forecast_objective(replay,source_train[-1][j],model.key)
                anchor=torch.stack([(p-ref[n]).square().mean()/(ref[n].square().mean()+1e-4)
                                    for n,p in model.decoder.named_parameters()]).mean()
                loss=loss+.01*anchor
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.)
            opt.step()
            total+=float(loss.detach())*len(i)/len(ix)
        sched.step()
        sv=score(model,source_val)
        tv=score(model,target_val) if adapt else None
        proxy=(.5*sv["score"]/bs["score"]+.5*tv["score"]/bt["score"]) if adapt else sv["score"]
        eligible=sv["MSE"] <= (1.1 if adapt else 1.03)*bs["MSE"]
        history.append({"round":r,"train_loss":total,"source":sv,"target_input":tv,"proxy":proxy,"eligible":eligible})
        if eligible and proxy<best_score:
            best,chosen,best_score=copy.deepcopy(model.state_dict()),r,proxy
        log(f"{'adapt' if adapt else 'source'} r={r} source={sv['MSE']:.6f} key={sv['key_MSE']:.6f} proxy={proxy:.6f} chosen={chosen}")
    model.load_state_dict(best)
    return model.eval(),{"selected_round":chosen,"score":best_score,"history":history}


def train_dataset(ds,args):
    data=prepare_peak(ds)
    cfg,src,tgt=data["cfg"],data["source"],data["target"]
    out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    out.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    kind="peak" if ds=="climate" else "residual"
    folder=MODEL_DIR/("peak_litetcn" if kind=="peak" else "residual_litetcn")/ds
    base_meta=json.loads((folder/"meta.json").read_text())
    spec={k:v for k,v in base_meta["model_spec"].items() if k not in ("architecture","shape_loss")} if kind=="peak" else {}
    manifest={"dataset":ds,"args":vars(args),"base_kind":kind,"base_spec":spec,"controls":CONTROLS,
        "source_selection":"minimum source validation 0.5global+0.5key MSE, global MSE guard +3%, zero correction fallback",
        "adaptation_selection":"equal relative source validation / within-target-input proxy, source MSE guard +10%, source fallback",
        "future_driver_inputs":"LiteTCN forecasts only, never actual future values",
        "target_future_labels_used_for_training_or_selection":False,
        "test_status":"reused development data; October 2026 was already inspected before this experiment",
        "code_sha256":{n:sha(Path(__file__).with_name(n)) for n in ("coupled_tcn.py","coupled_train.py","progressive_tcn.py","peak_model.py","peak_data.py","residual_tcn.py","model.py","data.py")},
        "data_sha256":{p.name:sha(p) for p in sorted((PROC_DIR/ds).glob("*.csv"))},
        "base_sha256":{str(p.relative_to(MODEL_DIR)):sha(p) for p in [folder/f"seed_{s}"/"source_model.pt" for s in args.seeds]}}
    manifest_path=out/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Run manifest changed")
    write_json(manifest_path,manifest)
    for seed in args.seeds:
        seed_dir=dest/f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        set_seed(seed)
        initial=CoupledLiteTCN(cfg,kind,spec)
        initial.base.load_state_dict(torch.load(folder/f"seed_{seed}"/"source_model.pt",weights_only=True))
        # Cached features depend only on inputs and frozen base predictions.
        st,sv=cache(initial,*src["train_packed"]),cache(initial,*src["val_packed"])
        tt,tv=cache_input(initial,tgt["train_packed"][0]),cache_input(initial,tgt["val_packed"][0])
        scores={}
        for control in CONTROLS:
            def log(message):
                text=f"{ds} seed={seed} control={control} {message}"
                print(text,flush=True)
                with (out/f"seed_{seed}.log").open("a",encoding="utf-8") as f:f.write(text+"\n")
            record=out/f"seed_{seed}_{control}.json"
            if record.exists() and (seed_dir/f"{control}.pt").exists():
                scores[control]=json.loads(record.read_text())["score"];continue
            model=copy.deepcopy(initial)
            model.set_control(control)
            model,history=fit(model,st,sv,None,None,seed,args,False,log)
            torch.save(model.state_dict(),seed_dir/f"{control}.pt")
            write_json(record,history)
            scores[control]=history["score"]
        chosen=min(scores,key=scores.get)
        model=CoupledLiteTCN(cfg,kind,spec,chosen)
        model.load_state_dict(torch.load(seed_dir/f"{chosen}.pt",weights_only=True))
        torch.save(model.state_dict(),seed_dir/"source_model.pt")
        record=out/f"seed_{seed}_adaptation.json"
        if record.exists() and (seed_dir/"adapted_model.pt").exists():
            history=json.loads(record.read_text())
        else:
            model,history=fit(model,st,sv,tt,tv,seed,args,True,log)
            torch.save(model.state_dict(),seed_dir/"adapted_model.pt")
            write_json(record,history)
        selection={"seed":seed,"source_control":chosen,"source_scores":scores,"adaptation_round":history["selected_round"],
            "adaptation_proxy":history["score"],"target_future_labels_used_for_training_or_selection":False}
        write_json(out/f"seed_{seed}_selection.json",selection)
        if seed==0:
            save_meta(ds,data,dest/"meta.json")
            meta=json.loads((dest/"meta.json").read_text())
            meta.update(model_spec={"architecture":FAMILY,"base_kind":kind,"base_spec":spec,"control":chosen},training_seed=0,
                adaptation="input_only_correction",experiment=FAMILY)
            write_json(dest/"meta.json",meta);write_json(dest/"selection.json",selection)
            for name in ("source_model","adapted_model"):(dest/f"{name}.pt").write_bytes((seed_dir/f"{name}.pt").read_bytes())
        del st,sv,tt,tv
    freeze={"timestamp_utc":pd.Timestamp.now(tz="UTC").isoformat(),"test_evaluated_at_freeze":False,
            "checkpoint_sha256":{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*.pt"))}}
    freeze_path=out/"selection_frozen.json"
    if freeze_path.exists():assert json.loads(freeze_path.read_text())["checkpoint_sha256"]==freeze["checkpoint_sha256"]
    else:write_json(freeze_path,freeze)


def evaluate_dataset(ds,seeds):
    data=prepare_peak(ds);cfg=data["cfg"]
    out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    manifest=json.loads((out/"manifest.json").read_text())
    rows,peaks,horizons,examples=[],[],[],{}
    for seed in seeds:
        for name in (*CONTROLS,"source_model","adapted_model"):
            model=CoupledLiteTCN(cfg,manifest["base_kind"],manifest["base_spec"])
            model.load_state_dict(torch.load(dest/f"seed_{seed}"/f"{name}.pt",weights_only=True))
            for role in ("source","target"):
                dom=data[role];x,y=dom["test_packed"]
                p=predict(model,x,bs=256)
                metrics,_=evaluate(ds,cfg,p,dom,fixed_scale(ds,role,dom["scaler"]))
                rows.append({"dataset":ds,"domain":role,"seed":seed,"method":name,**metrics})
                key=cfg["columns"].index(cfg["key"]);sd=dom["scaler"].std[key]
                peaks.append({"dataset":ds,"domain":role,"seed":seed,"method":name,**peak_metrics(y.numpy()[...,key]*sd,p[...,key]*sd)})
                for h in (1,3,6,12,24):
                    horizons.append({"dataset":ds,"domain":role,"seed":seed,"method":name,"horizon_hours":h,
                        "key_RMSE":float(np.sqrt(np.mean((p[:,:h,key]-y.numpy()[:,:h,key])**2))*sd)})
                if seed==0:
                    ix=np.linspace(0,len(x)-1,min(256,len(x)),dtype=int)
                    examples[f"{role}_{name}"]=p[ix]
                    examples[f"{role}_X"],examples[f"{role}_Y"]=(v.numpy()[ix] for v in dom["test"])
                    examples[f"{role}_indices"]=ix
    f=pd.DataFrame(rows);f.to_csv(out/"metrics_all.csv",index=False)
    pd.DataFrame(peaks).to_csv(out/"peak_metrics.csv",index=False)
    pd.DataFrame(horizons).to_csv(out/"horizon_metrics.csv",index=False)
    np.savez_compressed(out/"forecast_examples.npz",**examples)
    print(ds+" FROZEN TEST\n"+f.query("domain=='target'").groupby("method")[["MSE","key_RMSE"]].mean().to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",nargs="+",default=["climate","energy"])
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    p.add_argument("--rounds",type=int,default=16)
    p.add_argument("--adapt-rounds",type=int,default=6)
    p.add_argument("--max-windows",type=int,default=4096)
    args=p.parse_args();torch.set_num_threads(4)
    for ds in args.datasets:train_dataset(ds,args)
    for ds in args.datasets:evaluate_dataset(ds,args.seeds)


if __name__=="__main__":main()
