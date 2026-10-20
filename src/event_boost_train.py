"""Peak-aware boosted forecasting with source replay and strict within-input adaptation.

Source/adaptation choices for every dataset/seed freeze before test prediction.
Tests are previously inspected development periods; no independent claim is made.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import catboost
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import torch

from .config import MODEL_DIR,PROC_DIR,RESULTS_DIR
from .data import save_meta
from .diagnose_peaks import peak_metrics
from .event_boost import (VARIANTS,EventBoostForecaster,features,encode,decode,event_quality,input_pair)
from .improve import evaluate,fixed_scale,write_json
from .peak_data import prepare_peak
from .progressive_train import sha
from .train import predict

FAMILY="event_boost"
WEIGHTS=(0.,.5,1.)


def sample(data,count):
    ix=np.linspace(0,len(data[0])-1,min(count,len(data[0])),dtype=int)
    return data[0][ix],data[1][ix]


def prep(packed,truth,cfg):
    f,offset,scale=features(packed,cfg)
    key=cfg["columns"].index(cfg["key"])
    return {"X":f,"offset":offset,"scale":scale,"Y":truth[...,key],"packed":packed}


def labels(data,variant):return encode(data["Y"],data["offset"],data["scale"],variant)


def train_weights(data,variant):
    if variant=="trajectory":return np.ones(len(data["X"]))
    span=np.ptp((data["Y"]-data["offset"][:,None])/data["scale"][:,None],axis=1)
    return 1.+.5*np.clip(span,0,6)


def predict_tree(model,data,variant,mode,trees,h):
    raw=model.predict(data["X"],ntree_end=trees,thread_count=4)
    return decode(raw,data["offset"],data["scale"],variant,mode,h)


def candidate_rows(model,data,variant,trees,h,baseline=None):
    raw=model.predict(data["X"],ntree_end=trees,thread_count=4)
    for mode in (("mean",) if variant=="trajectory" else ("mean","extrema","phase_extrema")):
        t=decode(raw,data["offset"],data["scale"],variant,mode,h)
        for weight in ((1.,) if baseline is None else WEIGHTS):
            p=t if baseline is None else (1-weight)*baseline+weight*t
            yield {"variant":variant,"mode":mode,"weight":weight,"trees":trees},p


def base_predictions(model,packed):return predict(model,torch.as_tensor(packed,dtype=torch.float32),bs=256)


def make_tree(seed,args,iterations,learning_rate):
    return CatBoostRegressor(iterations=iterations,depth=4,border_count=32,learning_rate=learning_rate,
        loss_function="MultiRMSE",l2_leaf_reg=10,random_seed=seed,thread_count=args.threads,
        verbose=100,allow_writing_files=False,bootstrap_type="Bernoulli",subsample=.8)


def run_fit(ds,args):
    data=prepare_peak(ds);cfg=data["cfg"];src,tgt=data["source"],data["target"]
    out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
    out.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    kind="peak" if ds=="climate" else "residual"
    folder=MODEL_DIR/("peak_litetcn" if ds=="climate" else "residual_litetcn")/ds
    meta=json.loads((folder/"meta.json").read_text())
    spec={k:v for k,v in meta["model_spec"].items() if k not in ("architecture","shape_loss")} if kind=="peak" else {}
    manifest={"dataset":ds,"args":vars(args),"catboost":catboost.__version__,"base_kind":kind,"base_spec":spec,
        "variants":VARIANTS,"weights":WEIGHTS,"source_trees":[100,200,args.source_trees],
        "adaptation_trees":[0,100,200,args.adapt_trees],"depth":4,"border_count":32,
        "source_objective":"MultiRMSE trajectory; geometry DCT16 + extrema2 + circular peak/trough phase4, high-span window weights",
        "adaptation_objective":"continued source trees, target within-input geometry/trajectory loss + 0.25 source replay, L2 leaf regularization 10",
        "selection":"event quality = point MSE + .5 error at actual extrema + .25 max/min value error + .1 ramp MSE; equal source/target-input relative scores; source key MSE guard +5%",
        "validation":"source validation stride25; target validation first L origins purged then stride25, non-overlapping tail labels",
        "target_future_labels_used_for_training_or_selection":False,"cross_window_target_label_reconstruction":False,
        "test_status":"previously inspected development data; no independent confirmation",
        "code_sha256":{n:sha(Path(__file__).with_name(n)) for n in ("event_boost.py","event_boost_train.py","peak_model.py","peak_data.py","model.py","residual_tcn.py","data.py")},
        "data_sha256":{p.name:sha(p) for p in sorted((PROC_DIR/ds).glob("*.csv"))},
        "base_sha256":{str(p.relative_to(MODEL_DIR)):sha(p) for s in args.seeds for p in (folder/f"seed_{s}"/"source_model.pt",folder/f"seed_{s}"/"adapted_model.pt")}}
    path=out/"manifest.json"
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(manifest)):raise ValueError("Run settings changed")
    write_json(path,manifest)
    sx,sy=sample(tuple(t.numpy() for t in src["train_packed"]),args.max_windows)
    vx,vy=(t.numpy()[::25] for t in src["val_packed"])
    tx,_=sample((tgt["train_packed"][0].numpy(),np.arange(len(tgt["train_packed"][0]))),args.max_windows)
    tx,ty=input_pair(tx,cfg["H"],len(cfg["columns"]))
    tv=tgt["val_packed"][0].numpy()[cfg["L"]::25]
    tv,tyv=input_pair(tv,cfg["H"],len(cfg["columns"]))
    st,sv,tt,tv=prep(sx,sy,cfg),prep(vx,vy,cfg),prep(tx,ty,cfg),prep(tv,tyv,cfg)
    key=cfg["columns"].index(cfg["key"])
    for seed in args.seeds:
        seed_dir=dest/f"seed_{seed}";seed_dir.mkdir(exist_ok=True)
        choice_path=out/f"seed_{seed}_selection.json"
        if choice_path.exists():continue
        started=time.perf_counter()
        empty={"variant":"trajectory","mode":"mean","weight":0.,"trees":0}
        source_base=EventBoostForecaster(cfg,kind,spec,None,empty)
        source_base.base.load_state_dict(torch.load(folder/f"seed_{seed}"/"source_model.pt",weights_only=True))
        adapted_base=EventBoostForecaster(cfg,kind,spec,None,empty)
        adapted_base.base.load_state_dict(torch.load(folder/f"seed_{seed}"/"adapted_model.pt",weights_only=True))
        source_p=base_predictions(source_base,vx)[...,key]
        adapter_source_p=base_predictions(adapted_base,vx)[...,key]
        adapter_target_p=base_predictions(adapted_base,tv["packed"])[...,key]
        source_ref=event_quality(sv["Y"],source_p)
        adapter_s_ref=event_quality(sv["Y"],adapter_source_p)
        adapter_t_ref=event_quality(tv["Y"],adapter_target_p)
        best_source={**empty,"quality":source_ref,"eligible":True,"candidate":"source_backbone"}
        best_adapter={**empty,"source":adapter_s_ref,"target_input":adapter_t_ref,"proxy":1.,"eligible":True,"candidate":"adapted_backbone"}
        source_search=[best_source.copy()];adapter_search=[best_adapter.copy()]
        for variant in VARIANTS:
            print(f"{ds} seed={seed} variant={variant}: source boosting",flush=True)
            model=make_tree(seed,args,args.source_trees,.06)
            model.fit(st["X"],labels(st,variant),sample_weight=train_weights(st,variant))
            model.save_model(str(seed_dir/f"{variant}_source_full.cbm"))
            best_standalone=None
            for n in sorted(set((100,200,args.source_trees))):
                for config,p in candidate_rows(model,sv,variant,n,cfg["H"]):
                    q=event_quality(sv["Y"],p)
                    if best_standalone is None or q["score"]<best_standalone["quality"]["score"]:best_standalone={**config,"quality":q}
                for config,p in candidate_rows(model,sv,variant,n,cfg["H"],source_p):
                    q=event_quality(sv["Y"],p)
                    eligible=q["MSE"]<=1.05*source_ref["MSE"]
                    row={**config,"quality":q,"eligible":eligible,"candidate":"source_tree_blend"}
                    source_search.append(row)
                    if eligible and q["score"]<best_source["quality"]["score"]:best_source=row
            model.shrink(ntree_end=best_standalone["trees"])
            model.save_model(str(seed_dir/f"{variant}_source.cbm"))
            write_json(out/f"seed_{seed}_{variant}_source.json",best_standalone)
            # Keep an exact source-prefix control and continue boosting on inner-X pairs.
            combined_x=np.concatenate((tt["X"],st["X"]))
            combined_y=np.concatenate((labels(tt,variant),labels(st,variant)))
            weights=np.r_[train_weights(tt,variant),.25*len(tt["X"])/len(st["X"])*train_weights(st,variant)]
            print(f"{ds} seed={seed} variant={variant}: within-input adaptation",flush=True)
            adapter=make_tree(seed+1000,args,args.adapt_trees,.04)
            adapter.fit(combined_x,combined_y,sample_weight=weights,init_model=model)
            adapter.save_model(str(seed_dir/f"{variant}_adapted_full.cbm"))
            source_n=model.tree_count_
            for added in sorted(set((0,100,200,args.adapt_trees))):
                n=source_n+added
                source_predictions=list(candidate_rows(adapter,sv,variant,n,cfg["H"],adapter_source_p))
                target_predictions=list(candidate_rows(adapter,tv,variant,n,cfg["H"],adapter_target_p))
                for (config,ps),(_,pt) in zip(source_predictions,target_predictions):
                    qs,qt=event_quality(sv["Y"],ps),event_quality(tv["Y"],pt)
                    proxy=.5*qs["score"]/adapter_s_ref["score"]+.5*qt["score"]/adapter_t_ref["score"]
                    eligible=qs["MSE"]<=1.05*adapter_s_ref["MSE"]
                    row={**config,"added_trees":added,"source":qs,"target_input":qt,"proxy":proxy,"eligible":eligible,"candidate":"source_tree" if not added else "adapted_tree"}
                    adapter_search.append(row)
                    if eligible and proxy<best_adapter["proxy"]:best_adapter=row
        for role,choice,backbone in (("source",best_source,source_base),("adapted",best_adapter,adapted_base)):
            torch.save(backbone.base.state_dict(),seed_dir/f"{role}_backbone.pt")
            if choice["weight"]:
                # Source choices refer to untruncated source training prefixes.
                file=f"{choice['variant']}_source_full.cbm" if role=="source" else f"{choice['variant']}_adapted_full.cbm"
                selected=CatBoostRegressor();selected.load_model(str(seed_dir/file))
                selected.shrink(ntree_end=choice["trees"])
                selected.save_model(str(seed_dir/f"{role}_tree.cbm"))
        selection={"seed":seed,"source":best_source,"adapted":best_adapter,
            "target_future_labels_used_for_training_or_selection":False,"training_seconds":time.perf_counter()-started}
        write_json(choice_path,selection)
        write_json(out/f"seed_{seed}_source_search.json",source_search)
        write_json(out/f"seed_{seed}_adaptation_search.json",adapter_search)
        if seed==0:
            save_meta(ds,data,dest/"meta.json")
            meta=json.loads((dest/"meta.json").read_text())
            meta.update(model_spec={"architecture":FAMILY,"base_kind":kind,"base_spec":spec},training_seed=0,experiment=FAMILY,
                        adaptation="within_input_event_boost",selection=selection)
            write_json(dest/"meta.json",meta);write_json(dest/"selection.json",selection)
    freeze={"timestamp_utc":pd.Timestamp.now(tz="UTC").isoformat(),"test_evaluated_at_freeze":False,
        "model_sha256":{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}}
    path=out/"selection_frozen.json"
    if path.exists():assert json.loads(path.read_text())["model_sha256"]==freeze["model_sha256"]
    else:write_json(path,freeze)


def load_seed(ds,seed,role):
    folder=MODEL_DIR/FAMILY/ds
    meta=json.loads((folder/"meta.json").read_text())
    selection=json.loads((RESULTS_DIR/FAMILY/ds/f"seed_{seed}_selection.json").read_text())[role]
    tree=None
    if selection["weight"]:
        tree=CatBoostRegressor();tree.load_model(str(folder/f"seed_{seed}"/f"{role}_tree.cbm"))
    model=EventBoostForecaster(meta,meta["model_spec"]["base_kind"],meta["model_spec"]["base_spec"],tree,selection)
    model.base.load_state_dict(torch.load(folder/f"seed_{seed}"/f"{role}_backbone.pt",weights_only=True))
    return model.eval()


def run_evaluation(ds,args):
    data=prepare_peak(ds);cfg=data["cfg"];out=RESULTS_DIR/FAMILY/ds
    rows,peaks,horizons,examples=[],[],[],{}
    for seed in args.seeds:
        source,adapter=load_seed(ds,seed,"source"),load_seed(ds,seed,"adapted")
        baseline=copy.deepcopy(adapter);baseline.selection={**baseline.selection,"weight":0.}
        for role in ("source","target"):
            dom=data[role];x,y=dom["test_packed"]
            for name,model in (("source_model",source),("adapted_model",adapter),("retained_backbone",baseline)):
                p=predict(model,x,bs=256)
                scores,_=evaluate(ds,cfg,p,dom,fixed_scale(ds,role,dom["scaler"]))
                rows.append({"dataset":ds,"seed":seed,"domain":role,"method":name,**scores})
                key=cfg["columns"].index(cfg["key"]);sd=dom["scaler"].std[key]
                peaks.append({"dataset":ds,"seed":seed,"domain":role,"method":name,**peak_metrics(y.numpy()[...,key]*sd,p[...,key]*sd)})
                for h in (1,3,6,12,24):horizons.append({"dataset":ds,"seed":seed,"domain":role,"method":name,"horizon_hours":h,
                    "key_RMSE":float(np.sqrt(np.mean((p[:,:h,key]-y.numpy()[:,:h,key])**2))*sd)})
                if seed==0:
                    ix=np.linspace(0,len(x)-1,min(256,len(x)),dtype=int)
                    examples[f"{role}_{name}"]=p[ix];examples[f"{role}_X"],examples[f"{role}_Y"]=(v.numpy()[ix] for v in dom["test"])
                    examples[f"{role}_indices"]=ix
    f=pd.DataFrame(rows);f.to_csv(out/"metrics_all.csv",index=False)
    pd.DataFrame(peaks).to_csv(out/"peak_metrics.csv",index=False);pd.DataFrame(horizons).to_csv(out/"horizon_metrics.csv",index=False)
    np.savez_compressed(out/"forecast_examples.npz",**examples)
    print(ds+" FROZEN DEVELOPMENT EVALUATION\n"+f.query("domain=='target'").groupby("method")[["MSE","key_RMSE"]].mean().to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",nargs="+",choices=("climate","energy"),default=["climate","energy"])
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    p.add_argument("--source-trees",type=int,default=400);p.add_argument("--adapt-trees",type=int,default=400)
    p.add_argument("--max-windows",type=int,default=8000);p.add_argument("--threads",type=int,default=4)
    args=p.parse_args();torch.set_num_threads(args.threads)
    for ds in args.datasets:run_fit(ds,args)
    for ds in args.datasets:run_evaluation(ds,args)


if __name__=="__main__":main()
