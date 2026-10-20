"""Export gradual-unfreezing evidence, including negative controls and fresh weather."""
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, RESULTS_DIR, ROOT
from .data import prepare
from .diagnose_peaks import peak_metrics
from .inference import (forecast_batches, load_bundle, load_dlinear_bundle,
                        load_peak_bundle, load_progressive_bundle, load_residual_bundle,
                        load_strict_bundle)
from .peak_report import figure, table
from .progressive_train import FAMILY
from .strict_report import previous

OUT = RESULTS_DIR/FAMILY
FIG = OUT/"figures"


def target_rows(family, ds, method):
    f = pd.read_csv(RESULTS_DIR/family/ds/"metrics_all.csv")
    return f[(f.domain=="target")&(f.method==method)]


def forecasts_plot(cfg, arrays, predictions, path, meta_family=FAMILY):
    key = cfg["columns"].index(cfg["key"])
    meta = json.loads((MODEL_DIR/meta_family/cfg["dataset"] / "meta.json").read_text())
    sd, mean = meta["target_scaler"]["std"][key], meta["target_scaler"]["mean"][key]
    fig, axes = plt.subplots(3,1,figsize=(12,8),layout="constrained")
    for ax,index in zip(axes,(0,128,255)):
        ax.plot(np.arange(-96,0),arrays["target_X"][index,-96:,key]*sd+mean,color="#929aa5",label="Observed input")
        ax.plot(np.arange(cfg["H"]),arrays["target_Y"][index,:,key]*sd+mean,color="#161b22",lw=2,label="Actual future")
        for label,pred,color in predictions:
            ax.plot(np.arange(cfg["H"]),pred[index,:,key]*sd+mean,color=color,label=label)
        ax.axvline(-.5,color="#afb8c1",ls=":")
        ax.set_title(f"Fixed window index {arrays['target_indices'][index]}",fontsize=10)
        ax.set_ylabel(cfg["key_unit"],fontsize=9)
        ax.grid(alpha=.15)
    axes[0].legend(ncol=3,fontsize=8)
    axes[-1].set_xlabel("Hours relative to forecast origin")
    fig.savefig(path,dpi=150)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    FIG.mkdir(exist_ok=True)
    content, comparisons, full_peaks, all_horizons = [], [], [], []
    for ds in ("climate","energy"):
        cfg = {**DATASETS[ds], "dataset":ds}
        out = OUT/ds
        f = pd.read_csv(out/"metrics_all.csv")
        old_method = "selected_adapted" if ds=="climate" else "adapted_residual_tcn"
        old_family = "peak_litetcn" if ds=="climate" else "residual_litetcn"
        if ds=="climate":
            best_rows = target_rows("strict_context",ds,"strict_adapted")
        else:
            best_rows = previous(ds)
        entries = [("Original saved LiteTCN (fixed seed 0)", target_rows("improved",ds,"legacy_care_v2_seed0")),
                   ("Previous retained LiteTCN",target_rows(old_family,ds,old_method)),
                   ("Previous best overall-MSE family",best_rows),
                   ("New head-only source control",f.query("domain=='target' and method=='head_only'")),
                   ("New gradual-unfreezing source control",f.query("domain=='target' and method=='progressive'")),
                   ("New selected source",f.query("domain=='target' and method=='source_model'")),
                   ("New input-adapted LiteTCN-R",f.query("domain=='target' and method=='adapted_model'"))]
        if ds=="energy":
            entries.append(("Existing strict ensemble (temperature priority)",target_rows("strict_refined",ds,"strict_adapted")))
        summary = pd.DataFrame([{"model":label,"MSE_mean":d.MSE.mean(),"MSE_seed_SD":d.MSE.std(),
                    "key_RMSE_mean":d.key_RMSE.mean(),"key_RMSE_seed_SD":d.key_RMSE.std(),
                    "MASE_train_mean":d.MASE_train.mean()} for label,d in entries])
        comparisons.append(summary.assign(dataset=ds))
        content.append('<h2>'+html.escape(cfg["title"])+'</h2>'+table(summary))
        models = [("Original LiteTCN",load_bundle(ds)[1]["adapted_model"],"#b98146"),
                  ("Previous LiteTCN",(load_peak_bundle(ds) if ds=="climate" else load_residual_bundle(ds))[1]["adapted_model"],"#8e70aa"),
                  ("Previous hybrid" if ds=="climate" else "Previous DLinear",(load_strict_bundle(ds,True) if ds=="climate" else load_dlinear_bundle(ds))[1]["adapted_model"],"#197c75"),
                  ("New input-adapted LiteTCN-R",load_progressive_bundle(ds)[1]["adapted_model"],"#326fba")]
        arrays = np.load(out/"forecast_examples.npz")
        old_arrays = np.load(RESULTS_DIR/old_family/ds/"forecast_examples.npz")
        original_arrays = np.load(RESULTS_DIR/"improved"/ds/"forecast_examples.npz")
        np.testing.assert_allclose(arrays["target_Y"],old_arrays["target_Y"])
        predictions = [("Original LiteTCN",original_arrays["target_legacy_care_v2_seed0"],"#b98146"),
                       ("Previous LiteTCN",old_arrays[f"target_{old_method}"],"#8e70aa")]
        if ds=="climate":
            prior = np.load(RESULTS_DIR/"strict_context"/ds/"forecast_examples.npz")["target_strict_adapted"]
        else:
            prior = forecast_batches(models[2][1],arrays["target_X"])
        predictions += [(models[2][0],prior,"#197c75"),(models[3][0],arrays["target_adapted_model"],"#326fba")]
        path = FIG/f"{ds}_progressive_forecasts.png"
        forecasts_plot(cfg,arrays,predictions,path)
        content.append(figure(path,"Same fixed first/middle/last origins used in earlier comparisons; deployment seed 0. These examples are not chosen for the new model's performance. Peak errors remain visible."))
        data = prepare(ds)
        truth = data["target"]["test"][1].numpy()
        key = cfg["columns"].index(cfg["key"])
        sd = data["target"]["scaler"].std[key]
        if ds=="energy":
            models.append(("Existing strict ensemble",load_strict_bundle(ds)[1]["adapted_model"],"#626c78"))
        horizons, peaks = [], []
        for name,model,color in models:
            if ds=="climate" and name=="Previous LiteTCN":
                from .peak_data import prepare_peak
                x = prepare_peak(ds)["target"]["test_packed"][0]
            else:
                x = data["target"]["test"][0]
            p = forecast_batches(model,x)
            peaks.append({"dataset":ds,"model":name,**peak_metrics(truth[...,key]*sd,p[...,key]*sd)})
            for h in (1,3,6,12,24):
                horizons.append({"dataset":ds,"model":name,"horizon_hours":h,
                                 "key_RMSE":float(np.sqrt(np.mean((p[:,:h,key]-truth[:,:h,key])**2))*sd)})
        peak_frame, horizon_frame = pd.DataFrame(peaks),pd.DataFrame(horizons)
        full_peaks.append(peak_frame);all_horizons.append(horizon_frame)
        content.append('<h3>Deployment seed 0: peaks, troughs and timing</h3>'+table(peak_frame[["model","RMSE","MAE_at_actual_peak","MAE_at_actual_trough","mean_absolute_peak_timing_steps","ramp_MAE"]]))
        fig,ax = plt.subplots(figsize=(8,4.3),layout="constrained")
        for label,_,color in models:
            d = horizon_frame[horizon_frame.model==label]
            ax.plot(d.horizon_hours,d.key_RMSE,label=label,color=color,marker="o")
        ax.set_xlabel("Cumulative forecast horizon (hours)");ax.set_ylabel(cfg["key"]+" RMSE (degC)")
        ax.set_title("Deployment seed 0: short vs 24-hour error")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        path=FIG/f"{ds}_progressive_horizons.png"
        fig.savefig(path,dpi=150);plt.close(fig)
        content.append(figure(path,"Each point measures the first h outputs of the same 24-hour model, not a separately trained short-horizon model. A shorter horizon does not establish better 24-hour performance."))
        controls = []
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
        for name,color in (("head_only","#8e70aa"),("progressive","#326fba")):
            record=json.loads((out/f"seed_0_{name}.json").read_text())
            hist=record["history"]
            controls.append({"control":name,"selected_round":record["selected_round"],"source_val_score":record["score"]})
            axes[0].plot([r["round"] for r in hist],[r["score"] for r in hist],label=name,color=color)
        axes[0].set_title("Source-validation selection objective")
        history=json.loads((out/"seed_0_adaptation.json").read_text())["history"]
        axes[1].plot([r["round"] for r in history],[r["proxy"] for r in history],marker="o",color="#326fba")
        axes[1].set_title("Source + target-input adaptation proxy")
        for ax in axes: ax.set_xlabel("Sampled round");ax.grid(alpha=.2)
        axes[0].axvline(4.5,color="#b5bdc8",ls=":");axes[0].axvline(8.5,color="#b5bdc8",ls=":")
        axes[0].legend(fontsize=8)
        path=FIG/f"{ds}_progressive_selection.png"
        fig.savefig(path,dpi=150);plt.close(fig)
        content.append('<h3>Selection evidence: seed 0</h3>'+table(pd.DataFrame(controls))+figure(path,"Source schedule: head rounds 1-4, projection 5-8, encoder 9-24. Adapter: head 1-2, projection 3-4, encoder 5-8. These curves contain no target forecast labels."))
        source_summary=f[f.domain=="source"].groupby("method")[["MSE","key_RMSE","MASE_train"]].mean().reset_index()
        content.append('<details><summary>Domain 1 retention metrics</summary>'+table(source_summary)+'</details>')
    pd.concat(comparisons).to_csv(OUT/"comparison.csv",index=False)
    pd.concat(full_peaks).to_csv(OUT/"deployment_peak_comparison.csv",index=False)
    pd.concat(all_horizons).to_csv(OUT/"deployment_horizon_comparison.csv",index=False)
    fresh=OUT/"climate_oct2026"
    fresh_frame=pd.read_csv(fresh/"metrics_all.csv")
    fresh_summary=fresh_frame.groupby(["domain","method"])[["MSE","key_RMSE"]].mean().reset_index()
    fresh_arrays=np.load(fresh/"forecast_examples.npz")
    cfg={**DATASETS["climate"],"dataset":"climate"}
    fresh_path=FIG/"climate_oct2026_progressive_forecasts.png"
    forecasts_plot(cfg,fresh_arrays,[("Original LiteTCN",fresh_arrays["target_original_LiteTCN_seed0"],"#b98146"),
        ("Previous hybrid",fresh_arrays["target_previous_hybrid"],"#197c75"),
        ("New input-adapted LiteTCN-R",fresh_arrays["target_progressive_LiteTCN_R"],"#326fba")],fresh_path)
    drivers=pd.read_csv(OUT/"energy_drivers"/"source_validation_forecasts.csv")
    gain=100*(1-drivers.iloc[1].source_val_24h_RMSE/drivers.iloc[0].source_val_24h_RMSE)
    audit=json.loads((OUT/"artifact_audit.json").read_text())
    validation=(OUT/"validation.log").read_text(encoding="utf-8")
    assert "Ran 27 tests" in validation and validation.rstrip().endswith("OK")
    efficiency=pd.read_csv(OUT/"efficiency.csv")
    body='''<h1>LiteTCN-R: gradual unfreezing and forecast-driver analysis</h1>
<p>Supporting technical evidence, 19 October 2026. This is not the team's final assessment report.</p>
<h2>Outcome and limits</h2>
<p>All matched source runs and input-only adaptations were completed for climate and energy across seeds 0, 1 and 2. Gradual source unfreezing did not improve the source-validation selection objective over its head-only control. Selection retained initial/early-head checkpoints. Actual encoder updates were exercised during training, verified by tests; deployment selection was not forced to accept a worse source checkpoint.</p>
<p>The new adapted model does not beat the strongest existing overall-MSE family on either development benchmark. Its energy temperature RMSE is slightly lower than the previous retained LiteTCN-R, while multivariate MSE worsens. The existing strict ensemble still has lower energy temperature RMSE. Climate development performance trails LiteTCN v3 and the hybrid. These are negative/limited results, not a solved peak problem. Existing defaults and checkpoints are preserved; the experiment is separately selectable in the interface.</p>
<h2>Model, controlled budget and adaptation</h2>
<p>Start each seed from the previously source-fitted LiteTCN-R ridge-head checkpoint. Retain its three original strided convolutions, projection, RevIN and DLinear-initialized shared temporal skip. Freeze the skip and RevIN throughout. Compare head-only against head → projection → full encoder, with identical seeds, sampled source windows, objective and 24-round budget. Disable projection dropout in both controls to isolate trainability from stochastic feature noise. At most 4,096 windows per round and batch 256; these are sampled rounds, not 24 full dataset passes.</p>
<p>AdamW weight decay 0.0001, cosine decay; learning rates head 0.0003, projection 0.00003, encoder 0.00001. Source objective = 0.5 global standardized MSE + 0.5 key-variable MSE + 0.001 relative layerwise source anchor. Select the minimum source-validation objective, reject global-MSE degradation above 3%, and include round zero. Both controls are retained and evaluated after selection freeze.</p>
<p>Adapt the selected source model for eight sampled rounds, using half the source learning rates. Withhold the final 24 observed values inside each individual target input. Pad only its remaining prefix with its first observed row; the withheld suffix cannot enter normalization or prediction. No target forecasting Y tensors or cross-window target-label reconstruction are used. Adapter objective = target-input forecast objective + source replay objective + 0.01 relative layerwise anchor. Select 0.5 relative source-validation objective + 0.5 relative target-input objective, with a 10% source-MSE guard and source-model fallback. The cached historical training-period scalers are unchanged for comparability.</p>
<p>All six seed/dataset policies and checkpoints were frozen before any new test evaluation. Development tests have been inspected in earlier experiments, so they are not independent confirmation. No test-based selection across the old/new families is presented as a validation-selected procedure.</p>
'''+''.join(content)+'''<h2>Source-only energy driver check</h2>
<p>Use only ETTh1 source train/validation data. Compare direct 24-hour oil-temperature-change ridge forecasts from 168 past temperatures and deterministic calendar features against the same features plus recent past load levels, means and changes. Ridge is selected separately on source validation; no target data, future actual loads or test forecasts enter the analysis. These associations are not causal effects.</p>
'''+table(drivers)+f'<p>Past-load features reduce 24-hour source-validation RMSE by {gain:.3f}%. This small result does not establish that a complicated load decoder will fix the peaks. Source-validation tuning and reporting share the same period.</p>'+'''<h2>New 1–17 October 2026 climate follow-up</h2>
<p>All compared model files and policies were hashed before downloading this new period. The API request supplies seven preceding context days and ends on 17 October. Each domain provides 385 overlapping hourly forecast origins. No further model/policy selection uses this period. This short, single-season follow-up is limited evidence; overlapping origins are not 385 independent replications. Weather archive observations may be revised. No independent new energy period is available in the current benchmark.</p>
'''+table(fresh_summary)+figure(fresh_path,"New October period, fixed first/middle/last origins, deployment seed 0. Small overall-MSE differences do not imply better temperature peaks.")+'''<h2>Verification and efficiency</h2>
<p>All 27 tests passed, including 52 UI dataset/family/domain combinations. Stage-update tests verify real encoder/projection weight changes and a fixed linear/RevIN baseline. Input isolation tests verify withheld-tail and other-window gradients are zero. Artifact checks recompute selection, compare initial/checkpoint/data/code hashes, verify selected encoder drift, and recompute complete deployment seed-0 test MSE.</p>
'''+table(efficiency)+'<details><summary>Artifact audit</summary><pre>'+html.escape(json.dumps(audit,indent=2))+'''</pre></details>
<h2>Reproduce</h2><pre>python -m src.progressive_train
python -m src.energy_driver_analysis
python -m src.progressive_confirm
python -m src.check_progressive
python -m unittest discover -s tests -v
python -m src.progressive_report
streamlit run app/app.py</pre>
<p>Checkpoints: models/progressive_litetcn. Metrics, selection histories, manifests and plots: results/progressive_litetcn. Finance models/results are unchanged.</p>'''
    document='<!doctype html><html><head><meta charset="utf-8"><title>LiteTCN gradual unfreezing results</title><style>body{max-width:1140px;margin:36px auto;padding:0 24px;font:16px/1.55 system-ui;color:#17212b}h1,h2,h3{line-height:1.2}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}th,td{padding:8px;border-bottom:1px solid #d6dce3;text-align:left}img{width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:13px;color:#56616e}pre{background:#f2f5f8;padding:16px;overflow:auto}details{margin:16px 0}</style></head><body>'+body+'</body></html>'
    path=ROOT/"report"/"IE643_Progressive_LiteTCN_Results.html"
    path.write_text(document,encoding="utf-8")
    print(path)
    print(pd.concat(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
