"""Final notes for source/retained-backbone driver decoder ablations."""
import html
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, RESULTS_DIR, ROOT
from .inference import load_coupled_bundle
from .peak_report import figure, table
from .progressive_report import forecasts_plot
from .strict_report import previous

OUT=RESULTS_DIR/"coupled_transfer"
FIG=OUT/"figures"


def read_rows(family,ds,method):
    f=pd.read_csv(RESULTS_DIR/family/ds/"metrics_all.csv")
    return f[(f.domain=="target")&(f.method==method)]


def main():
    torch.set_num_threads(2)
    FIG.mkdir(exist_ok=True)
    body,comparisons,headline,efficiency=[],[],[],[]
    for ds in ("climate","energy"):
        cfg={**DATASETS[ds],"dataset":ds}
        prior_family="peak_litetcn" if ds=="climate" else "residual_litetcn"
        prior_method="selected_adapted" if ds=="climate" else "adapted_residual_tcn"
        old=read_rows(prior_family,ds,prior_method)
        new=read_rows("coupled_transfer",ds,"adapted_model")
        best=read_rows("strict_context",ds,"strict_adapted") if ds=="climate" else previous(ds)
        entries=[("Original saved LiteTCN (fixed seed 0)",read_rows("improved",ds,"legacy_care_v2_seed0")),
            ("Previous retained LiteTCN",old),("Previous best overall-MSE family",best),
            ("Gradual unfreezing / input adaptation",read_rows("progressive_litetcn",ds,"adapted_model")),
            ("Source-backbone decoder / input adaptation",read_rows("coupled_litetcn",ds,"adapted_model")),
            ("Retained adapted backbone + source decoder",read_rows("coupled_transfer",ds,"source_decoder_on_adapted_backbone")),
            ("Retained adapted backbone + input-adapted decoder",new)]
        if ds=="energy":entries.append(("Existing strict ensemble (temperature priority)",read_rows("strict_refined",ds,"strict_adapted")))
        summary=pd.DataFrame([{"model":name,"MSE_mean":d.MSE.mean(),"MSE_seed_SD":d.MSE.std(),
            "key_RMSE_mean":d.key_RMSE.mean(),"key_RMSE_seed_SD":d.key_RMSE.std(),"MASE_train_mean":d.MASE_train.mean()} for name,d in entries])
        comparisons.append(summary.assign(dataset=ds))
        headline.append({"dataset":ds,"prior_retained_TCN_MSE":old.MSE.mean(),"new_TCN_MSE":new.MSE.mean(),
            "MSE_gain_pct":100*(1-new.MSE.mean()/old.MSE.mean()),"prior_retained_key_RMSE":old.key_RMSE.mean(),
            "new_key_RMSE":new.key_RMSE.mean(),"key_RMSE_gain_pct":100*(1-new.key_RMSE.mean()/old.key_RMSE.mean()),
            "best_overall_family_MSE":best.MSE.mean()})
        body.append('<h2>'+html.escape(cfg["title"])+'</h2>'+table(summary))
        arrays=np.load(OUT/ds/"forecast_examples.npz")
        old_arrays=np.load(RESULTS_DIR/prior_family/ds/"forecast_examples.npz")
        np.testing.assert_allclose(arrays["target_Y"],old_arrays["target_Y"])
        original=np.load(RESULTS_DIR/"improved"/ds/"forecast_examples.npz")
        predictions=[("Original LiteTCN",original["target_legacy_care_v2_seed0"],"#b98146"),
            ("Previous retained LiteTCN",old_arrays[f"target_{prior_method}"],"#8e70aa"),
            ("New decoder on adapted LiteTCN",arrays["target_adapted_model"],"#326fba")]
        if ds=="climate":
            hybrid=np.load(RESULTS_DIR/"strict_context"/ds/"forecast_examples.npz")["target_strict_adapted"]
            predictions.insert(2,("Previous hybrid",hybrid,"#197c75"))
        else:
            ensemble=np.load(RESULTS_DIR/"strict_refined"/ds/"forecast_examples.npz")["target_strict_adapted"]
            predictions.insert(2,("Existing strict ensemble",ensemble,"#197c75"))
        path=FIG/f"{ds}_final_tcn_forecasts.png"
        forecasts_plot(cfg,arrays,predictions,path,meta_family="coupled_transfer")
        body.append(figure(path,"Same fixed first/middle/last deployment seed-0 examples. The purple and blue lines nearly overlap because the incremental correction is small. Black is actual future; no peak problem is claimed solved."))
        peaks=pd.read_csv(OUT/ds/"peak_metrics.csv")
        peaks=peaks[(peaks.domain=="target")&(peaks.method.isin(("adapted_backbone_only","adapted_model")))]
        peak_summary=peaks.groupby("method")[["RMSE","MAE_at_actual_peak","MAE_at_actual_trough","mean_absolute_peak_timing_steps","ramp_MAE"]].mean().reset_index()
        body.append('<h3>Peak/trough diagnostics: means of three training seeds</h3>'+table(peak_summary))
        horizon=pd.read_csv(OUT/ds/"horizon_metrics.csv")
        horizon=horizon[horizon.domain=="target"].groupby(["method","horizon_hours"]).key_RMSE.mean().reset_index()
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
        for name,label,color in (("adapted_backbone_only","Retained backbone","#8e70aa"),("adapted_model","With input-adapted decoder","#326fba")):
            d=horizon[horizon.method==name]
            axes[0].plot(d.horizon_hours,d.key_RMSE,marker="o",label=label,color=color)
        axes[0].set_title("Cumulative temperature error (three-seed mean)")
        axes[0].set_ylabel("RMSE (degC)");axes[0].set_xlabel("Hours forecast")
        history=json.loads((OUT/ds/"seed_0_selection.json").read_text())["history"]
        axes[1].plot(range(len(history)),[r["proxy"] for r in history],marker="o",color="#326fba")
        axes[1].set_title("Source + target-input selection proxy (seed 0)")
        axes[1].set_xticks(range(len(history)),[r["candidate"].replace("input_round_","r") for r in history],rotation=45,ha="right",fontsize=8)
        axes[1].set_ylabel("Relative objective")
        for ax in axes:ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        path=FIG/f"{ds}_final_tcn_horizons.png";fig.savefig(path,dpi=150);plt.close(fig)
        body.append(figure(path,"Short-horizon points use prefixes of the same 24-hour forecast, not separately trained models. Selection curves use source validation and individual target input tails only."))
        controls=[]
        for seed in (0,1,2):
            old_selection=json.loads((RESULTS_DIR/"coupled_litetcn"/ds/f"seed_{seed}_selection.json").read_text())
            new_selection=json.loads((OUT/ds/f"seed_{seed}_selection.json").read_text())
            controls.append({"seed":seed,"source_control":old_selection["source_control"],"source_scores":str(old_selection["source_scores"]),"composition_choice":new_selection["chosen"]})
        body.append('<details><summary>Control selection: all seeds</summary>'+table(pd.DataFrame(controls))+'</details>')
        for family in ("coupled_litetcn","coupled_transfer"):
            f=pd.read_csv(RESULTS_DIR/family/ds/"metrics_all.csv")
            body.append('<details><summary>'+family+' source/target metrics, including negative controls</summary>'+table(f.groupby(["domain","method"])[["MSE","key_RMSE","MASE_train"]].mean().reset_index())+'</details>')
        meta,models=load_coupled_bundle(ds,True);model=models["adapted_model"]
        from .peak_model import pack_calendar
        # Latency uses a real-shape input with synthetic deterministic timestamps;
        # values do not affect the operation graph. This is not an accuracy check.
        sample=torch.from_numpy(pack_calendar(arrays["target_X"][:1],pd.date_range("2024-01-01",periods=cfg["L"],freq="h").to_numpy()[None]))
        torch.set_num_threads(1)
        with torch.no_grad():
            for _ in range(5):model(sample)
            started=time.perf_counter()
            for _ in range(30):model(sample)
        efficiency.append({"dataset":ds,"parameters":sum(p.numel() for p in model.parameters()),"updated_decoder_parameters":sum(p.numel() for p in model.decoder.parameters()),
            "batch_1_CPU_ms":(time.perf_counter()-started)/30*1000,"threads":1,"warmups":5,"repeats":30})
        torch.set_num_threads(2)
    headline=pd.DataFrame(headline);headline.to_csv(OUT/"headline.csv",index=False)
    pd.concat(comparisons).to_csv(OUT/"comparison.csv",index=False)
    pd.DataFrame(efficiency).to_csv(OUT/"efficiency.csv",index=False)
    validation=(OUT/"validation.log").read_text(encoding="utf-8")
    assert "Ran 29 tests" in validation and validation.rstrip().endswith("OK")
    audit=json.loads((OUT/"artifact_audit.json").read_text())
    first_audit=json.loads((RESULTS_DIR/"coupled_litetcn"/"artifact_audit.json").read_text())
    energy_drivers=pd.read_csv(RESULTS_DIR/"progressive_litetcn"/"energy_drivers"/"source_validation_forecasts.csv")
    climate_drivers=pd.read_csv(RESULTS_DIR/"coupled_litetcn"/"climate_drivers"/"source_validation_forecasts.csv")
    document='''<h1>LiteTCN follow-up: unfreezing and predicted-driver correction</h1>
<p>Supporting model evidence · 19 October 2026 · not the final assessment report. All old checkpoints are preserved.</p>
<h2>Measured conclusion</h2>
<p>The final composition yields a small energy improvement over retained LiteTCN-R: 0.146226 → 0.146098 multivariate MSE and 4.09915 → 4.08351 degC temperature RMSE. Each of the three energy training seeds improves slightly. Climate is effectively unchanged/slightly worse: 0.220609 → 0.220615 MSE, 1.32844 → 1.32935 degC. These results do not establish a meaningful peak fix or a new best model. The previous hybrid/DLinear alternatives remain stronger overall; the existing strict ensemble remains stronger for energy temperature RMSE. Training-seed consistency is not statistical significance over independent time periods.</p>
'''+table(headline)+'''<h2>What was tested</h2>
<ol><li>Gradual LiteTCN-R encoder unfreezing: matched head-only control, three seeds, 24 sampled rounds. Source validation rejects late unfreezing checkpoints. Input adaptation gives limited/mixed development results; see the separate gradual-unfreezing notes.</li>
<li>Source-only driver analysis: compare temperature/calendar forecasts with and without past load or weather summaries. No target data or true future driver inputs. These are predictive-association diagnostics, not causal claims.</li>
<li>Small decoder on frozen source LiteTCNs: compare temperature-only vs driver-conditioned residuals using the same multivariate base and decoder size. The temperature-only ablation removes extra non-temperature decoder conditioning; the base itself is still multivariate. Source validation selects coupled conditioning for climate, temperature-only for energy. Source-validation gains do not reliably transfer.</li>
<li>Compose the decoder with the retained input-adapted LiteTCN. This avoids comparing a source-only backbone with a fully adapted earlier model. Include exact zero correction, source decoder and input-adapted decoder candidates; preserve the backbone weights and every non-temperature forecast exactly.</li></ol>
<h2>Architecture and optimization</h2>
<p>The base forecasts all channels: LiteTCN v3 for climate, LiteTCN-R for energy. The new temperature head has 2,689 parameters, two 32-unit GELU layers and a zero-initialized scalar output. Inputs at each future step are the base's predicted variables, their changes relative to the last observation, observed-window means/log standard deviations/last normalized levels/recent means, deterministic future calendar features, and relative horizon. All normalization statistics use the observed input only. Multiply the head output by the observed temperature standard deviation and add it only to the temperature forecast. No observed future load, cloud, humidity or radiation values are supplied.</p>
<p>Decoder source controls use AdamW 0.0005, weight decay 0.001, cosine decay, 16 sampled rounds of at most 4,096 windows, batch 256. Objective and source selection use 0.5 global standardized MSE + 0.5 temperature MSE, with a 3% source-MSE guard and zero-correction fallback. Adaptation of the source-backbone decoder uses six rounds at 0.0001, source replay and relative-layer anchor 0.01. Final composition uses eight rounds at 0.0001, the same replay/anchor, and both zero/source-head initial candidates. Models are fitted for seeds 0, 1 and 2 on CPU.</p>
<p>Target adaptation withholds the final 24 observed values inside each individual input. Prefix padding never contains the withheld suffix. All decoder features are built from that prefix and the frozen backbone's forecasts. Target forecasting Y tensors and reconstructed labels from neighboring windows are excluded. Final composition selection is 0.5 relative source-validation objective + 0.5 relative target-input objective, with a source-MSE guard of 10% relative to the zero-correction backbone. Cached historical training-period scalers remain unchanged for comparison. All dataset/seed choices are frozen before test inference.</p>
<p>Development tests have already been inspected in earlier work. The 1–17 October weather check was frozen before download for the gradual-unfreezing model, but was already inspected before these decoder experiments. It is not claimed as independent confirmation for either decoder model; no decoder result is selected using that period. No independent new energy period is available.</p>
'''+''.join(body)+'''<h2>Driver diagnostics: source-validation evidence only</h2>
<p>Separate source-fitted ridge models compare all 168 past temperatures plus known calendar features against the same features with past driver levels, means and changes. Ridge strength is selected on source validation; reporting reuses that validation, so it is not independent confirmation. Past energy loads reduce 24-hour RMSE by about 0.19%; past climate weather measurements reduce it by about 2.77%. Stronger causal/physical conclusions are not established.</p><h3>Energy</h3>
'''+table(energy_drivers)+'<h3>Climate</h3>'+table(climate_drivers)+'''<h2>Verification</h2>
<p>All 29 tests passed, including 56 UI dataset/family/domain combinations. Tests check actual staged encoder updates, fixed skip/normalizer/base weights, exact zero-initialized decoder behavior, unchanged non-temperature forecasts, the decoder ablation mask, input-tail isolation, timestamps, serialization and source/target CSV inference. Artifact audits recompute selections, verify frozen/source/base/data/code hashes, and recompute complete deployment seed-0 test MSE.</p>
'''+table(pd.DataFrame(efficiency))+'<details><summary>Composition audit</summary><pre>'+html.escape(json.dumps(audit,indent=2))+'</pre></details><details><summary>Source-backbone control audit</summary><pre>'+html.escape(json.dumps(first_audit,indent=2))+'''</pre></details>
<h2>Files and reproduction</h2>
<pre>python -m src.progressive_train
python -m src.energy_driver_analysis
python -m src.climate_driver_analysis
python -m src.coupled_train
python -m src.coupled_transfer
python -m src.check_progressive
python -m src.check_coupled
python -m src.check_coupled_transfer
python -m unittest discover -s tests -v
python -m src.coupled_report
streamlit run app/app.py</pre>
<p>Source-backbone controls: models/results/coupled_litetcn. Final composed checkpoints: models/coupled_transfer. Composition metrics, histories, manifests, audit and plots: results/coupled_transfer. Finance results remain unchanged. Existing interface defaults are preserved; choose LiteTCN predicted-driver decoder to inspect the new composition. Supporting notes do not replace the required official-template student report.</p>'''
    path=ROOT/"report"/"IE643_Coupled_LiteTCN_Results.html"
    path.write_text('<!doctype html><html><head><meta charset="utf-8"><title>LiteTCN predicted-driver follow-up</title><style>body{max-width:1140px;margin:36px auto;padding:0 24px;font:16px/1.55 system-ui;color:#17212b}h1,h2,h3{line-height:1.2}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}th,td{padding:8px;border-bottom:1px solid #d6dce3;text-align:left}img{width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:13px;color:#56616e}pre{background:#f2f5f8;padding:16px;overflow:auto}details{margin:16px 0}</style></head><body>'+document+'</body></html>',encoding="utf-8")
    print(path);print(headline.to_string(index=False))


if __name__=="__main__":main()
