"""Source/target next-hour LiteTCN training, separate from 24-hour forecasts.

Rolling evaluation uses new observed history at every origin. It is never labelled
as a single 24-hour forecast. Target fitting sees individual X windows only.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import MODEL_DIR,PROC_DIR,RESULTS_DIR
from .data import save_meta
from .diagnose_peaks import peak_metrics
from .improve import evaluate,fixed_scale,write_json
from .metrics import seasonal_naive
from .nowcast_tcn import NextHourTCN,within_input,turning_weights,forecast_loss
from .peak_data import prepare_peak
from .train import predict,set_seed
from .progressive_train import sha

FAMILY="next_hour"
CONTROLS=("uniform","turning")


@torch.no_grad()
def quality(model,x,y,key):
    p=predict(model,x,bs=256)
    error=(p-y.numpy())**2
    w=turning_weights(x,y,key).numpy()
    mse,kmse=float(error.mean()),float(error[...,key].mean())
    turning=float(np.mean(error[:,0,key]*w)/np.mean(w))
    return {"MSE":mse,"key_MSE":kmse,"turning_weighted_key_MSE":turning,"score":.5*mse+.25*kmse+.25*turning}


def source_fit(train,val,cfg,seed,control,args,log):
    set_seed(seed)
    gen=torch.Generator().manual_seed(seed)
    model=NextHourTCN(cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.rounds,eta_min=1e-5)
    key=cfg["columns"].index(cfg["key"])
    baseline=quality(model,*val,key)
    best,selected,score=copy.deepcopy(model.state_dict()),0,baseline["score"]
    hist=[{"round":0,**baseline,"eligible":True}]
    for r in range(1,args.rounds+1):
        model.train();ix=torch.randperm(len(train[0]),generator=gen)[:args.max_windows]
        total=0.
        for i in ix.split(256):
            x,y=train[0][i],train[1][i]
            loss=forecast_loss(model(x),y,x,key,control=="turning")
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(i)/len(ix)
        sched.step();q=quality(model,*val,key)
        eligible=q["MSE"]<=1.05*baseline["MSE"]
        hist.append({"round":r,"train_loss":total,**q,"eligible":eligible})
        if eligible and q["score"]<score:best,selected,score=copy.deepcopy(model.state_dict()),r,q["score"]
        log(f"{control} r={r} source_val={q['MSE']:.6f} key={q['key_MSE']:.6f} selected={selected}")
    model.load_state_dict(best)
    return model.eval(),{"selected_round":selected,"score":score,"history":hist}


def adapt_fit(source,train,val,target_train_x,target_val_x,cfg,seed,args,log):
    set_seed(seed+1000);gen=torch.Generator().manual_seed(seed+1000)
    model=copy.deepcopy(source)
    ref={n:p.detach().clone() for n,p in source.named_parameters()}
    opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.adapt_rounds,eta_min=1e-5)
    key=cfg["columns"].index(cfg["key"])
    xt,yt=within_input(target_train_x,model.C)
    xv,yv=within_input(target_val_x,model.C)
    bs,bt=quality(model,*val,key),quality(model,xv,yv,key)
    best,selected,best_proxy=copy.deepcopy(model.state_dict()),0,1.
    history=[{"round":0,"source":bs,"target_input":bt,"proxy":1.,"eligible":True}]
    for r in range(1,args.adapt_rounds+1):
        model.train();ix=torch.randperm(len(xt),generator=gen)[:args.max_windows]
        si=torch.randint(len(train[0]),(len(ix),),generator=gen)
        for i,j in zip(ix.split(256),si.split(256)):
            anchor=torch.stack([(p-ref[n]).square().mean()/(ref[n].square().mean()+1e-4) for n,p in model.named_parameters()]).mean()
            loss=forecast_loss(model(xt[i]),yt[i],xt[i],key,True)+.5*forecast_loss(model(train[0][j]),train[1][j],train[0][j],key,True)+.001*anchor
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        sched.step();sv,tv=quality(model,*val,key),quality(model,xv,yv,key)
        proxy=.5*sv["score"]/bs["score"]+.5*tv["score"]/bt["score"]
        eligible=sv["MSE"]<=1.1*bs["MSE"] and sv["key_MSE"]<=1.1*bs["key_MSE"]
        history.append({"round":r,"source":sv,"target_input":tv,"proxy":proxy,"eligible":eligible})
        if eligible and proxy<best_proxy:best,selected,best_proxy=copy.deepcopy(model.state_dict()),r,proxy
        log(f"adapt r={r} source_val={sv['MSE']:.6f} input={tv['MSE']:.6f} proxy={proxy:.6f} selected={selected}")
    model.load_state_dict(best)
    return model.eval(),{"selected_round":selected,"proxy":best_proxy,"history":history,
                         "target_future_labels_used_for_training_or_selection":False}


def run_fit(ds,args):
    data=prepare_peak(ds);cfg={**data["cfg"],"H":1}
    src,tgt=data["source"],data["target"]
    out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    out.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    manifest={"dataset":ds,"args":vars(args),"context":48,"horizon":1,"controls":CONTROLS,
        "source_loss":".5 global MSE + .5 key MSE, optionally 3x weight at observed direction reversals",
        "adaptation_loss":"target withheld-last-hour loss + .5 source replay + .001 relative layer source anchor",
        "selection":"equal relative source validation / target input scores, 10% source global/key guard and epoch-zero fallback",
        "target_future_labels_used_for_training_or_selection":False,
        "evaluation":"next-hour rolling predictions with observations updated hourly; not a single 24-hour forecast",
        "test_status":"reused development periods; no independent confirmation",
        "code_sha256":{n:sha(Path(__file__).with_name(n)) for n in ("nowcast_tcn.py","nowcast_train.py","peak_data.py","peak_model.py","data.py")},
        "data_sha256":{p.name:sha(p) for p in sorted((PROC_DIR/ds).glob("*.csv"))}}
    path=out/"manifest.json"
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(manifest)):raise ValueError("Run changed")
    write_json(path,manifest)
    train=(src["train_packed"][0],src["train_packed"][1][:,:1])
    val=(src["val_packed"][0][::25],src["val_packed"][1][::25,:1])
    tv=tgt["val_packed"][0][cfg["L"]::25]
    for seed in args.seeds:
        started=time.perf_counter();seed_dir=dest/f"seed_{seed}";seed_dir.mkdir(exist_ok=True)
        def log(text):
            text=f"{ds} seed={seed} {text}";print(text,flush=True)
            with (out/f"seed_{seed}.log").open("a",encoding="utf-8") as f:f.write(text+"\n")
        scores={}
        for control in CONTROLS:
            record=out/f"seed_{seed}_{control}.json"
            if record.exists() and (seed_dir/f"{control}.pt").exists():
                scores[control]=json.loads(record.read_text())["score"];continue
            model,history=source_fit(train,val,cfg,seed,control,args,log)
            torch.save(model.state_dict(),seed_dir/f"{control}.pt");write_json(record,history);scores[control]=history["score"]
        chosen=min(scores,key=scores.get)
        model=NextHourTCN(cfg);model.load_state_dict(torch.load(seed_dir/f"{chosen}.pt",weights_only=True))
        torch.save(model.state_dict(),seed_dir/"source_model.pt")
        record=out/f"seed_{seed}_adaptation.json"
        if record.exists() and (seed_dir/"adapted_model.pt").exists():history=json.loads(record.read_text())
        else:
            model,history=adapt_fit(model,train,val,tgt["train_packed"][0],tv,cfg,seed,args,log)
            torch.save(model.state_dict(),seed_dir/"adapted_model.pt");write_json(record,history)
        selection={"seed":seed,"source_control":chosen,"source_scores":scores,"adaptation_round":history["selected_round"],
            "adaptation_proxy":history["proxy"],"target_future_labels_used_for_training_or_selection":False,
            "seconds_this_invocation":time.perf_counter()-started}
        write_json(out/f"seed_{seed}_selection.json",selection)
        if seed==0:
            save_meta(ds,{**data,"cfg":cfg},dest/"meta.json")
            meta=json.loads((dest/"meta.json").read_text());meta.update(model_spec={"architecture":FAMILY},training_seed=0,
                experiment=FAMILY,adaptation="turning_aware_input_only",forecast_semantics="one future hour, refreshed observations at each rolling origin")
            write_json(dest/"meta.json",meta);write_json(dest/"selection.json",selection)
            for name in ("source_model","adapted_model"):(dest/f"{name}.pt").write_bytes((seed_dir/f"{name}.pt").read_bytes())
    freeze={"timestamp_utc":pd.Timestamp.now(tz="UTC").isoformat(),"test_evaluated_at_freeze":False,
        "checkpoint_sha256":{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*.pt"))}}
    path=out/"selection_frozen.json"
    if path.exists():assert json.loads(path.read_text())["checkpoint_sha256"]==freeze["checkpoint_sha256"]
    else:write_json(path,freeze)


def evaluate_dataset(ds,args):
    data=prepare_peak(ds);cfg={**data["cfg"],"H":1}
    out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    rows,peaks,examples=[],[],{}
    for seed in args.seeds:
        models={}
        for name in ("source_model","adapted_model"):
            model=NextHourTCN(cfg);model.load_state_dict(torch.load(dest/f"seed_{seed}"/f"{name}.pt",weights_only=True));models[name]=model.eval()
        for role in ("source","target"):
            dom=data[role];x,y=dom["test_packed"];truth=y[:,:1]
            reduced={**dom,"test":(dom["test"][0],truth)}
            forecasts={name:predict(m,x,bs=256) for name,m in models.items()}
            forecasts["persistence"]=dom["test"][0][:,-1:].numpy()
            forecasts["seasonal_naive"]=dom["test"][0][:,-cfg["season"]:-cfg["season"]+1].numpy()
            for name,p in forecasts.items():
                metrics,_=evaluate(ds,cfg,p,reduced,fixed_scale(ds,role,dom["scaler"]))
                rows.append({"dataset":ds,"seed":seed,"domain":role,"method":name,**metrics})
                # Consecutive hourly origins form a rolling 24-hour episode;
                # observations are refreshed at EVERY point in this block.
                key=cfg["columns"].index(cfg["key"]);sd=dom["scaler"].std[key]
                n=(len(p)//24)*24
                y24=truth[:n,0,key].numpy().reshape(-1,24)*sd
                p24=p[:n,0,key].reshape(-1,24)*sd
                peaks.append({"dataset":ds,"seed":seed,"domain":role,"method":name,
                    "semantics":"rolling 1h forecasts, refreshed hourly",**peak_metrics(y24,p24)})
                if seed==0:examples[f"{role}_{name}"]=p
            if seed==0:
                examples[f"{role}_truth"]=truth.numpy();examples[f"{role}_X"]=dom["test"][0].numpy()
    frame=pd.DataFrame(rows);frame.to_csv(out/"metrics_all.csv",index=False)
    pd.DataFrame(peaks).to_csv(out/"rolling_peak_metrics.csv",index=False)
    np.savez_compressed(out/"rolling_forecasts.npz",**examples)
    print(ds+" NEXT-HOUR FROZEN EVALUATION\n"+frame.query("domain=='target'").groupby("method")[["MSE","key_RMSE"]].mean().to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",nargs="+",choices=("climate","energy"),default=["climate","energy"])
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2]);p.add_argument("--rounds",type=int,default=28)
    p.add_argument("--adapt-rounds",type=int,default=12);p.add_argument("--max-windows",type=int,default=4096)
    p.add_argument("--threads",type=int,default=2);args=p.parse_args();torch.set_num_threads(args.threads)
    for ds in args.datasets:run_fit(ds,args)
    for ds in args.datasets:evaluate_dataset(ds,args)


if __name__=="__main__":main()
