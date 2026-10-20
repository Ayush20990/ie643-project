"""Post-freeze ablation of added target-input trees, holding calibration fixed."""
import copy
import json

import numpy as np
import pandas as pd
import torch

from .config import RESULTS_DIR
from .diagnose_peaks import peak_metrics
from .event_boost_train import load_seed
from .improve import evaluate,fixed_scale
from .peak_data import prepare_peak
from .train import predict


def main():
    torch.set_num_threads(2)
    for ds in ("climate","energy"):
        out=RESULTS_DIR/"event_boost"/ds
        assert (out/"selection_frozen.json").exists()
        data=prepare_peak(ds);cfg=data["cfg"];rows,peaks=[],[]
        for seed in (0,1,2):
            adapted=load_seed(ds,seed,"adapted")
            control=copy.deepcopy(adapted)
            if control.selection["weight"]:
                info=json.loads((out/f"seed_{seed}_{control.selection['variant']}_source.json").read_text())
                control.selection={**control.selection,"trees":info["trees"]}
            for role in ("source","target"):
                dom=data[role];x,y=dom["test_packed"]
                p=predict(control,x,bs=256)
                m,_=evaluate(ds,cfg,p,dom,fixed_scale(ds,role,dom["scaler"]))
                rows.append({"dataset":ds,"seed":seed,"domain":role,"method":"no_new_boost_adaptation",**m})
                key=cfg["columns"].index(cfg["key"]);sd=dom["scaler"].std[key]
                peaks.append({"dataset":ds,"seed":seed,"domain":role,"method":"no_new_boost_adaptation",
                    **peak_metrics(y.numpy()[...,key]*sd,p[...,key]*sd)})
        pd.DataFrame(rows).to_csv(out/"boost_adapter_ablation.csv",index=False)
        pd.DataFrame(peaks).to_csv(out/"boost_adapter_peak_ablation.csv",index=False)
        print(ds+" added-tree ablation\n"+pd.DataFrame(rows).query("domain=='target'")[["MSE","key_RMSE"]].mean().to_string(),flush=True)


if __name__=="__main__":main()
