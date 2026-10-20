"""Compose the source-fitted decoder with the existing input-adapted LiteTCN.

This controls for the frozen-source-backbone limitation in coupled_train. The
retained adapted backbone is an explicit zero-correction fallback. Development
and October data have already been inspected; no independent claim is made.
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
from .coupled_train import cache, cache_input, score
from .data import save_meta
from .diagnose_peaks import peak_metrics
from .improve import evaluate, fixed_scale, write_json
from .peak_data import prepare_peak
from .progressive_tcn import forecast_objective
from .progressive_train import sha
from .train import predict, set_seed

FAMILY="coupled_transfer"


def fit(model,st,sv,tt,tv,seed,args,log):
    set_seed(seed+3000)
    gen=torch.Generator().manual_seed(seed+3000)
    zero=copy.deepcopy(model)
    with torch.no_grad():
        zero.decoder[-1].weight.zero_();zero.decoder[-1].bias.zero_()
    baseline_s,baseline_t=score(zero,sv),score(zero,tv)
    def snapshot(m,name):
        s,t=score(m,sv),score(m,tv)
        proxy=.5*s["score"]/baseline_s["score"]+.5*t["score"]/baseline_t["score"]
        return {"candidate":name,"source":s,"target_input":t,"proxy":proxy,"eligible":s["MSE"]<=1.1*baseline_s["MSE"]}
    history=[snapshot(zero,"zero_correction"),snapshot(model,"source_decoder")]
    best_row=min((r for r in history if r["eligible"]),key=lambda r:r["proxy"])
    if best_row["candidate"]=="zero_correction":model=copy.deepcopy(zero)
    best=copy.deepcopy(model.state_dict());chosen=best_row["candidate"];best_proxy=best_row["proxy"]
    ref={n:p.detach().clone() for n,p in model.decoder.named_parameters()}
    opt=torch.optim.AdamW(model.decoder.parameters(),lr=1e-4,weight_decay=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.rounds,eta_min=1e-6)
    for r in range(1,args.rounds+1):
        ix=torch.randperm(len(tt[-1]),generator=gen)[:args.max_windows]
        si=torch.randint(len(st[-1]),(len(ix),),generator=gen)
        for i,j in zip(ix.split(256),si.split(256)):
            target=model.from_features(*(v[i] for v in tt[:3]))
            source=model.from_features(*(v[j] for v in st[:3]))
            anchor=torch.stack([(p-ref[n]).square().mean()/(ref[n].square().mean()+1e-4) for n,p in model.decoder.named_parameters()]).mean()
            loss=forecast_objective(target,tt[-1][i],model.key)+forecast_objective(source,st[-1][j],model.key)+.01*anchor
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.);opt.step()
        sched.step()
        row=snapshot(model,f"input_round_{r}");history.append(row)
        if row["eligible"] and row["proxy"]<best_proxy:
            best,chosen,best_proxy=copy.deepcopy(model.state_dict()),row["candidate"],row["proxy"]
        log(f"r={r} source={row['source']['MSE']:.6f} input={row['target_input']['MSE']:.6f} proxy={row['proxy']:.6f} selected={chosen}")
    model.load_state_dict(best)
    return model.eval(),zero.eval(),{"chosen":chosen,"proxy":best_proxy,"history":history,
        "target_future_labels_used_for_training_or_selection":False}


def train_dataset(ds,args):
    data=prepare_peak(ds);cfg=data["cfg"];src,tgt=data["source"],data["target"]
    out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    out.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    stage1=MODEL_DIR/"coupled_litetcn"/ds
    meta=json.loads((stage1/"meta.json").read_text());spec={k:v for k,v in meta["model_spec"].items() if k!="architecture"}
    base_folder=MODEL_DIR/("peak_litetcn" if ds=="climate" else "residual_litetcn")/ds
    manifest={"dataset":ds,"args":vars(args),"model_spec":spec,"backbone":"retained input-adapted LiteTCN v3 (climate) / R (energy)",
        "selection":"equal relative source validation / target input proxy, 10% source MSE guard relative to zero correction",
        "target_future_labels_used_for_training_or_selection":False,"test_status":"reused development; October previously inspected",
        "code_sha256":{n:sha(Path(__file__).with_name(n)) for n in ("coupled_transfer.py","coupled_tcn.py","coupled_train.py","peak_model.py","peak_data.py","progressive_tcn.py","model.py","residual_tcn.py","data.py")},
        "data_sha256":{p.name:sha(p) for p in sorted((PROC_DIR/ds).glob("*.csv"))},
        "initial_checkpoint_sha256":{str(p.relative_to(MODEL_DIR)):sha(p) for s in args.seeds for p in
            (stage1/f"seed_{s}"/"source_model.pt",base_folder/f"seed_{s}"/"adapted_model.pt")}}
    manifest_path=out/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=json.loads(json.dumps(manifest)):raise ValueError("Run changed")
    write_json(manifest_path,manifest)
    for seed in args.seeds:
        seed_dir=dest/f"seed_{seed}";seed_dir.mkdir(exist_ok=True)
        source_state=torch.load(stage1/f"seed_{seed}"/"source_model.pt",weights_only=True)
        torch.save(source_state,seed_dir/"source_model.pt")
        record=out/f"seed_{seed}_selection.json"
        if record.exists() and (seed_dir/"adapted_model.pt").exists():selection=json.loads(record.read_text())
        else:
            model=CoupledLiteTCN(cfg,**spec);model.load_state_dict(source_state)
            model.base.load_state_dict(torch.load(base_folder/f"seed_{seed}"/"adapted_model.pt",weights_only=True))
            torch.save(model.state_dict(),seed_dir/"source_decoder_on_adapted_backbone.pt")
            st,sv=cache(model,*src["train_packed"]),cache(model,*src["val_packed"])
            tt,tv=cache_input(model,tgt["train_packed"][0]),cache_input(model,tgt["val_packed"][0])
            def log(text):
                text=f"{ds} seed={seed} {text}";print(text,flush=True)
                with (out/f"seed_{seed}.log").open("a",encoding="utf-8") as f:f.write(text+"\n")
            model,zero,selection=fit(model,st,sv,tt,tv,seed,args,log)
            torch.save(model.state_dict(),seed_dir/"adapted_model.pt")
            torch.save(zero.state_dict(),seed_dir/"adapted_backbone_only.pt")
            write_json(record,selection);del st,sv,tt,tv
        if seed==0:
            save_meta(ds,data,dest/"meta.json")
            meta=json.loads((dest/"meta.json").read_text())
            meta.update(model_spec={"architecture":FAMILY,**spec},training_seed=0,adaptation="input_only_driver_correction",experiment=FAMILY)
            write_json(dest/"meta.json",meta);write_json(dest/"selection.json",selection)
            for name in ("source_model","adapted_model"):(dest/f"{name}.pt").write_bytes((seed_dir/f"{name}.pt").read_bytes())
    freeze={"timestamp_utc":pd.Timestamp.now(tz="UTC").isoformat(),"test_evaluated_at_freeze":False,
        "checkpoint_sha256":{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*.pt"))}}
    path=out/"selection_frozen.json"
    if path.exists():assert json.loads(path.read_text())["checkpoint_sha256"]==freeze["checkpoint_sha256"]
    else:write_json(path,freeze)


def evaluate_dataset(ds,seeds):
    data=prepare_peak(ds);cfg=data["cfg"]
    out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    spec=json.loads((out/"manifest.json").read_text())["model_spec"]
    rows,peaks,horizons,examples=[],[],[],{}
    for seed in seeds:
        for name in ("source_model","adapted_backbone_only","source_decoder_on_adapted_backbone","adapted_model"):
            model=CoupledLiteTCN(cfg,**spec)
            model.load_state_dict(torch.load(dest/f"seed_{seed}"/f"{name}.pt",weights_only=True))
            for role in ("source","target"):
                dom=data[role];x,y=dom["test_packed"];p=predict(model,x,bs=256)
                metrics,_=evaluate(ds,cfg,p,dom,fixed_scale(ds,role,dom["scaler"]))
                rows.append({"dataset":ds,"domain":role,"seed":seed,"method":name,**metrics})
                key=cfg["columns"].index(cfg["key"]);sd=dom["scaler"].std[key]
                peaks.append({"dataset":ds,"domain":role,"seed":seed,"method":name,**peak_metrics(y.numpy()[...,key]*sd,p[...,key]*sd)})
                for h in (1,3,6,12,24):
                    horizons.append({"dataset":ds,"domain":role,"seed":seed,"method":name,"horizon_hours":h,
                        "key_RMSE":float(np.sqrt(np.mean((p[:,:h,key]-y.numpy()[:,:h,key])**2))*sd)})
                if seed==0:
                    ix=np.linspace(0,len(x)-1,min(256,len(x)),dtype=int)
                    examples[f"{role}_{name}"]=p[ix];examples[f"{role}_X"],examples[f"{role}_Y"]=(v.numpy()[ix] for v in dom["test"])
                    examples[f"{role}_indices"]=ix
    f=pd.DataFrame(rows);f.to_csv(out/"metrics_all.csv",index=False)
    pd.DataFrame(peaks).to_csv(out/"peak_metrics.csv",index=False);pd.DataFrame(horizons).to_csv(out/"horizon_metrics.csv",index=False)
    np.savez_compressed(out/"forecast_examples.npz",**examples)
    print(ds+" FROZEN TEST\n"+f.query("domain=='target'").groupby("method")[["MSE","key_RMSE"]].mean().to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",nargs="+",default=["climate","energy"]);p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    p.add_argument("--rounds",type=int,default=8);p.add_argument("--max-windows",type=int,default=4096)
    args=p.parse_args();torch.set_num_threads(4)
    for ds in args.datasets:train_dataset(ds,args)
    for ds in args.datasets:evaluate_dataset(ds,args.seeds)


if __name__=="__main__":main()
