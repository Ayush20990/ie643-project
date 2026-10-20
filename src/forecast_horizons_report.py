"""Separate rolling one-hour and single-origin 24-hour forecasting evidence."""
import argparse
import html
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import DATASETS,MODEL_DIR,RESULTS_DIR,ROOT
from .context_models import ContextEnsemble
from .data import prepare
from .diagnose_peaks import peak_metrics
from .horizon_reference import FirstHourReference
from .improve import evaluate,fixed_scale
from .inference import load_bundle
from .peak_report import figure,table
from .progressive_report import forecasts_plot
from .strict_models import StrictEnsemble
from .strict_report import previous
from .train import predict

OUT=RESULTS_DIR/"forecast_horizons"
FIG=OUT/"figures"


def overview_plot():
    fig,axes=plt.subplots(2,1,figsize=(11,6),layout="constrained")
    for ax,ds in zip(axes,("climate","energy")):
        cfg=DATASETS[ds];key=cfg["columns"].index(cfg["key"])
        a=np.load(RESULTS_DIR/"peak_next_hour"/ds/"rolling_forecasts.npz")
        meta=json.loads((MODEL_DIR/"peak_next_hour"/ds/"meta.json").read_text())
        sd,mean=meta["target_scaler"]["std"][key],meta["target_scaler"]["mean"][key]
        for name,label,color,style in (("truth","Actual","#151a22","-"),("source_model","Source-only","#c48141","--"),("adapted_model","Adapted","#2674b9","-")):
            ax.plot(np.arange(96),a[f"target_{name}"][:96,0,key]*sd+mean,label=label,color=color,ls=style,lw=1.8)
        ax.set_title(f"{ds.title()}: first 96 hourly forecast origins",fontsize=11)
        ax.set_ylabel("Temperature (degC)");ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=9);axes[-1].set_xlabel("Hours (observations refreshed before every next-hour forecast)")
    path=FIG/"rolling_next_hour_overview.png";fig.savefig(path,dpi=120);plt.close(fig)
    return path


def rolling_plot(ds,arrays,path,meta_family="peak_next_hour"):
    cfg=DATASETS[ds];key=cfg["columns"].index(cfg["key"])
    meta=json.loads((MODEL_DIR/meta_family/ds/"meta.json").read_text())
    sd,mean=meta["target_scaler"]["std"][key],meta["target_scaler"]["mean"][key]
    n=len(arrays["target_truth"])
    starts=(0,n//2,n-96)
    fig,axes=plt.subplots(3,1,figsize=(12,8.5),layout="constrained")
    for ax,start in zip(axes,starts):
        i=slice(start,start+96);t=np.arange(96)
        ax.plot(t,arrays["target_truth"][i,0,key]*sd+mean,color="#151a22",lw=2,label="Actual")
        previous=arrays["target_X"][i,-1,key]*sd+mean
        ax.plot(t,previous,color="#969da5",lw=1,alpha=.7,label="Persistence rolling 1h")
        ax.plot(t,arrays["target_source_model"][i,0,key]*sd+mean,color="#c48141",lw=1.4,ls="--",label="Source-only rolling 1h")
        ax.plot(t,arrays["target_adapted_model"][i,0,key]*sd+mean,color="#2674b9",lw=1.7,label="Adapted rolling 1h")
        ax.set_title(f"Fixed 96-hour segment starting at forecast-origin index {start}",fontsize=10)
        ax.set_ylabel(cfg["key_unit"],fontsize=9);ax.grid(alpha=.2)
    axes[0].legend(fontsize=9,ncol=3)
    axes[-1].set_xlabel("Hours in rolling evaluation (observations updated before every prediction)")
    fig.suptitle("Rolling next-hour forecasts — observed history refreshed every hour",fontsize=13)
    fig.savefig(path,dpi=150);plt.close(fig)
    return starts


def matched_references(ds):
    path=OUT/f"{ds}_matched_reference_metrics.csv"
    peaks_path=OUT/f"{ds}_matched_reference_peaks.csv"
    if path.exists():return pd.read_csv(path),pd.read_csv(peaks_path)
    data=prepare(ds);cfg={**data["cfg"],"H":1}
    family="strict_context" if ds=="climate" else "strict_refined"
    rows,peaks=[],[]
    for seed in (0,1,2):
        record=json.loads((RESULTS_DIR/family/ds/f"seed_{seed}_selection.json").read_text())
        spec=record["model_spec"]["neural_spec"]
        for role in ("source","target"):
            dom=data[role];truth=dom["test"][1][:,:1]
            for name in ("source_model","adapted_model"):
                base=(ContextEnsemble if ds=="climate" else StrictEnsemble)(data["cfg"],spec)
                base.load_state_dict(torch.load(MODEL_DIR/family/ds/f"seed_{seed}"/f"{name}.pt",weights_only=True))
                p=predict(FirstHourReference(base),dom["test"][0],bs=256)
                m,_=evaluate(ds,cfg,p,{**dom,"test":(dom["test"][0],truth)},fixed_scale(ds,role,dom["scaler"]))
                rows.append({"dataset":ds,"seed":seed,"domain":role,"method":name,"family":"Earlier hybrid, first-hour output",**m})
                key=cfg["columns"].index(cfg["key"]);sd=dom["scaler"].std[key];n=(len(p)//24)*24
                peaks.append({"dataset":ds,"seed":seed,"domain":role,"method":name,"family":"Earlier hybrid, first-hour output",
                    **peak_metrics(truth[:n,0,key].numpy().reshape(-1,24)*sd,p[:n,0,key].reshape(-1,24)*sd)})
            if seed==0:
                legacy=load_bundle(ds)[1]
                for name in ("source_model","adapted_model"):
                    p=predict(FirstHourReference(legacy[name]),dom["test"][0],bs=256)
                    m,_=evaluate(ds,cfg,p,{**dom,"test":(dom["test"][0],truth)},fixed_scale(ds,role,dom["scaler"]))
                    rows.append({"dataset":ds,"seed":0,"domain":role,"method":name,"family":"Original LiteTCN first-hour (fixed seed 0)",**m})
    pd.DataFrame(rows).to_csv(path,index=False);pd.DataFrame(peaks).to_csv(peaks_path,index=False)
    return pd.DataFrame(rows),pd.DataFrame(peaks)


def build_rolling(ds):
    cfg=DATASETS[ds]
    hybrid=pd.read_csv(RESULTS_DIR/"next_hour_hybrid"/ds/"metrics_all.csv")
    calibrated=pd.read_csv(RESULTS_DIR/"peak_next_hour"/ds/"metrics_all.csv")
    neural=pd.read_csv(RESULTS_DIR/"next_hour"/ds/"metrics_all.csv")
    pieces=[calibrated.assign(family="Peak-calibrated next-hour hybrid"),hybrid.assign(family="Next-hour LiteTCN + local dynamics"),neural.assign(family="Next-hour LiteTCN")]
    reference,reference_peaks=matched_references(ds)
    pieces.append(reference)
    all_metrics=pd.concat(pieces,ignore_index=True)
    summary=all_metrics.groupby(["domain","family","method"])[["MSE","key_RMSE","MASE_train"]].mean().reset_index()
    summary.to_csv(OUT/f"{ds}_next_hour_comparison.csv",index=False)
    tgt=calibrated[calibrated.domain=="target"]
    adapted=tgt[tgt.method=="adapted_model"];source=tgt[tgt.method=="source_model"]
    naive=neural[(neural.domain=="target")&(neural.method=="persistence")]
    headline={"dataset":ds,"family":"Peak-calibrated next-hour hybrid","horizon_hours":1,"source_MSE":source.MSE.mean(),"adapted_MSE":adapted.MSE.mean(),
        "source_key_RMSE":source.key_RMSE.mean(),"adapted_key_RMSE":adapted.key_RMSE.mean(),"persistence_key_RMSE":naive.key_RMSE.mean(),
        "key_RMSE_gain_from_adaptation_pct":100*(1-adapted.key_RMSE.mean()/source.key_RMSE.mean()),
        "key_RMSE_gain_vs_persistence_pct":100*(1-adapted.key_RMSE.mean()/naive.key_RMSE.mean()),
        "MSE_gain_vs_persistence_pct":100*(1-adapted.MSE.mean()/naive.MSE.mean())}
    arrays=np.load(RESULTS_DIR/"peak_next_hour"/ds/"rolling_forecasts.npz")
    path=FIG/f"{ds}_rolling_next_hour.png";starts=rolling_plot(ds,arrays,path)
    pd.DataFrame({"segment_start_origin":starts,"length_hours":96,"selection":"fixed first/middle/last, no performance filtering"}).to_csv(OUT/f"{ds}_rolling_plot_origins.csv",index=False)
    body='<h2>'+html.escape(cfg["title"])+': rolling next-hour task</h2>'+figure(path,
        "Deployment seed 0. Each point is a new one-hour forecast based on the observations available at that point; model weights stay frozen. Fixed first/middle/last segments are not chosen by forecast error.")
    body+='<h3>Same-horizon comparison</h3>'+table(summary[summary.domain=="target"])
    body+='<p>The earlier hybrid reference reuses the first output of its 24-hour head; it was not retrained as a next-hour model. It remains a strong climate reference and must not be omitted when judging the new model. The original saved LiteTCN is a fixed seed-0 reference; other neural families average three training seeds.</p>'
    peak_frames=[pd.read_csv(RESULTS_DIR/"peak_next_hour"/ds/"rolling_peak_metrics.csv").assign(family="Peak-calibrated next-hour hybrid"),
                 pd.read_csv(RESULTS_DIR/"next_hour_hybrid"/ds/"rolling_peak_metrics.csv").assign(family="Next-hour hybrid"),
                 pd.read_csv(RESULTS_DIR/"next_hour"/ds/"rolling_peak_metrics.csv").assign(family="Next-hour TCN/baselines"),reference_peaks]
    peaks=pd.concat(peak_frames,ignore_index=True)
    peaks=peaks[peaks.domain=="target"].groupby(["family","method"])[["RMSE","MAE_at_actual_peak","MAE_at_actual_trough","mean_absolute_peak_timing_steps","ramp_MAE"]].mean().reset_index()
    peaks.to_csv(OUT/f"{ds}_next_hour_peak_comparison.csv",index=False)
    body+='<h3>Peak/dip errors over non-overlapping rolling 24-hour episodes</h3>'+table(peaks)
    body+='<p>Episode metrics group 24 consecutive one-hour predictions, each with refreshed observations. These are not forecasts issued together for the next 24 hours. The final incomplete episode is omitted from these event metrics; pointwise metrics above include every original benchmark origin.</p>'
    body+='<p>The final calibration improves peak and trough errors versus the first next-hour hybrid on both datasets. It does not dominate every reference: the earlier climate hybrid has slightly lower peak MAE, and energy persistence still has lower trough MAE. Energy peak calibration also has slightly higher overall temperature RMSE than the first next-hour hybrid. These tradeoffs are retained in the tables, rather than selecting models after seeing the test errors.</p>'
    body+='<details><summary>Domain 1 and source-retention metrics</summary>'+table(summary[summary.domain=="source"])+'</details>'
    calibration=json.loads((RESULTS_DIR/"peak_next_hour"/ds/"seed_0_selection.json").read_text())["calibration"]
    key=cfg["columns"].index(cfg["key"]);record=calibration[key]
    cm=json.loads((MODEL_DIR/"peak_next_hour"/ds/"meta.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
    for ax,role,label in zip(axes,("source","target_input"),("Source observed validation episodes","Target observed-input episodes")):
        reference=record[f"{role}_point_reference"]
        adapted=record[f"adapted_{role}_point"]
        sd=cm["source_scaler" if role=="source" else "target_scaler"]["std"][key]
        ax.bar(("Source mixture","Adapted mixture"),np.sqrt([reference,adapted])*sd,color=("#c48141","#2674b9"))
        ax.set_title(label,fontsize=10);ax.set_ylabel("Temperature RMSE (degC)");ax.grid(axis="y",alpha=.2)
    path=FIG/f"{ds}_peak_calibration.png";fig.savefig(path,dpi=150);plt.close(fig)
    body+=figure(path,"Final calibration, deployment seed 0: all points being scored are inside the observed validation histories. The source-retention guard allows a small source error increase in exchange for target-input adaptation; no target test labels enter this fit.")
    history=json.loads((RESULTS_DIR/"next_hour"/ds/"seed_0_adaptation.json").read_text())["history"]
    meta=json.loads((MODEL_DIR/"next_hour"/ds/"meta.json").read_text());key=cfg["columns"].index(cfg["key"])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    axes[0].plot([r["round"] for r in history],[r["source"]["key_MSE"]**.5*meta["source_scaler"]["std"][key] for r in history],label="Source validation",color="#c48141")
    axes[0].plot([r["round"] for r in history],[r["target_input"]["key_MSE"]**.5*meta["target_scaler"]["std"][key] for r in history],label="Withheld target-input hour",color="#2674b9")
    axes[0].set_ylabel("Temperature RMSE (degC)");axes[0].legend(fontsize=8)
    axes[1].plot([r["round"] for r in history],[r["proxy"] for r in history],marker="o",color="#2674b9")
    axes[1].set_ylabel("Source / input selection proxy")
    for ax in axes:ax.set_xlabel("Adaptation round");ax.grid(alpha=.2)
    path=FIG/f"{ds}_next_hour_adaptation.png";fig.savefig(path,dpi=150);plt.close(fig)
    body+=figure(path,"These curves contain source validation and target observed-input pretexts only, not target future-test labels. The final hybrid adds a source-anchored local dynamics fit and a guarded per-variable mixture.")
    return body,headline


def build_long(ds):
    cfg={**DATASETS[ds],"dataset":ds}
    f=pd.read_csv(RESULTS_DIR/"event_boost"/ds/"metrics_all.csv")
    summary=f.groupby(["domain","method"])[["MSE","key_RMSE","MASE_train"]].mean().reset_index()
    a=np.load(RESULTS_DIR/"event_boost"/ds/"forecast_examples.npz")
    old=np.load(RESULTS_DIR/"improved"/ds/"forecast_examples.npz")
    predictions=[("Original LiteTCN",old["target_legacy_care_v2_seed0"],"#b98146"),
                 ("Retained adapted LiteTCN",a["target_retained_backbone"],"#8e70aa"),
                 ("Peak-aware boosted adaptation",a["target_adapted_model"],"#2674b9")]
    path=FIG/f"{ds}_single_origin_24h.png"
    forecasts_plot(cfg,a,predictions,path,meta_family="event_boost")
    body='<h2>'+html.escape(cfg["title"])+': single-origin 24-hour task</h2>'+table(summary)
    if ds=="climate":
        prior=pd.read_csv(RESULTS_DIR/"strict_context"/ds/"metrics_all.csv")
        prior=prior.query("domain=='target' and method=='strict_adapted'")
    else:prior=previous(ds)
    refs=[{"model":"Earlier strongest overall-MSE family","MSE":prior.MSE.mean(),"key_RMSE":prior.key_RMSE.mean()},
          {"model":"New boosted adapted model","MSE":f.query("domain=='target' and method=='adapted_model'").MSE.mean(),
           "key_RMSE":f.query("domain=='target' and method=='adapted_model'").key_RMSE.mean()}]
    body+='<h3>Earlier strongest overall reference retained</h3>'+table(pd.DataFrame(refs))
    body+='<p>The new boosted experiment modestly improves the retained TCN, but does not beat the earlier strongest overall family. Its 24-hour peak errors remain substantially larger than rolling one-hour errors; shortening the task must not be described as fixing the original 24-hour problem.</p>'
    body+=figure(path,"Same fixed first/middle/last origins as earlier 24-hour comparisons. Every colored forecast uses only the history before hour 0. There are no observation updates inside this forecast horizon.")
    p=pd.read_csv(RESULTS_DIR/"event_boost"/ds/"peak_metrics.csv")
    p=p[p.domain=="target"].groupby("method")[["RMSE","MAE_at_actual_peak","MAE_at_actual_trough","mean_predicted_range","mean_actual_range","mean_absolute_peak_timing_steps"]].mean().reset_index()
    body+='<h3>24-hour peak and timing diagnostics</h3>'+table(p)
    ablation_path=RESULTS_DIR/"event_boost"/ds/"boost_adapter_ablation.csv"
    if ablation_path.exists():
        control=pd.read_csv(ablation_path)
        body+='<h3>Added-tree adaptation ablation</h3>'+table(control.groupby("domain")[["MSE","key_RMSE"]].mean().reset_index())
        body+='<p>This control retains the selected backbone, postprocessing and mixture weight but removes the new target-input boosting stages. It isolates the added-tree contribution; it is not a separately reselected deployment policy.</p>'
    baseline=f.query("domain=='target' and method=='retained_backbone'")
    adapted=f.query("domain=='target' and method=='adapted_model'")
    return body,{"dataset":ds,"horizon_hours":24,"retained_MSE":baseline.MSE.mean(),"adapted_MSE":adapted.MSE.mean(),
        "retained_key_RMSE":baseline.key_RMSE.mean(),"adapted_key_RMSE":adapted.key_RMSE.mean(),
        "key_RMSE_gain_vs_retained_pct":100*(1-adapted.key_RMSE.mean()/baseline.key_RMSE.mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--rolling-only",action="store_true");args=p.parse_args()
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    content,rolling_headlines=[],[]
    for ds in ("climate","energy"):
        body,headline=build_rolling(ds);content.append(body);rolling_headlines.append(headline)
    rolling=pd.DataFrame(rolling_headlines);rolling.to_csv(OUT/"next_hour_headline.csv",index=False)
    overview=overview_plot()
    if args.rolling_only:
        print(rolling.to_string(index=False));return
    long_content,long_headlines=[],[]
    for ds in ("climate","energy"):
        body,headline=build_long(ds);long_content.append(body);long_headlines.append(headline)
    long=pd.DataFrame(long_headlines);long.to_csv(OUT/"twenty_four_hour_headline.csv",index=False)
    audit=json.loads((OUT/"artifact_audit.json").read_text())
    efficiency=pd.read_csv(OUT/"efficiency.csv")
    validation=(OUT/"validation.log").read_text(encoding="utf-8")
    count=re.search(r"Ran (\d+) tests",validation).group(1)
    assert re.search(r"\nOK\s*(?:UI combinations checked: \d+\s*)?$",validation)
    ui_count=re.search(r"UI combinations checked: (\d+)",validation).group(1)
    finance=pd.read_csv(RESULTS_DIR/"strict_refined"/"finance"/"metrics_all.csv")
    finance=finance[finance.method.isin(("strict_source","strict_adapted","persistence","input_mean"))].groupby(["domain","method"])[["MSE","key_RMSE","MASE_train"]].mean().reset_index()
    document='''<h1>Forecast horizon, peak tracking and label-free adaptation</h1>
<p>Supporting experimental evidence · 19 October 2026 · Ayush Patel (23B0037), Rishik Yadav (23B2103). This is not the team's final assessment report.</p>
<h2>Two explicitly different forecasting tasks</h2>
<p>The project statement does not fix a forecast horizon. The user explicitly approved adding rolling next-hour results while retaining the 24-hour benchmark. A rolling one-hour forecast receives a newly observed history at each hour. A single-origin 24-hour forecast must predict all 24 values without those observations. Lower one-hour error must not be presented as a like-for-like improvement over a 24-hour error.</p>
<p>The new next-hour source/adapted models and guarded TCN/dynamics hybrid track peaks much more closely at the shorter approved horizon. Adaptation and persistence comparisons below use the same one-hour task. Earlier hybrids are also evaluated using their first forecast output; they remain strong references, particularly in climate. No claim is made that every new model beats every older model.</p>
'''+table(rolling)+'''<h2>One-hour methodology</h2>
<p>LiteTCN is retained as a small unstrided causal residual TCN: width 32, depthwise/pointwise blocks with dilations 1, 2 and 4, observed 48-hour normalization/context and deterministic calendar features. It predicts the next-hour increment for all seven variables, starting exactly at persistence with a zero-initialized head. Source controls compare ordinary vs turning-weighted loss with identical seed and training budgets.</p>
<p>The source loss is 0.5 global standardized MSE + 0.5 key-variable MSE; the turning control gives 3x weight to temperature direction reversals. Source checkpoint selection balances global, key and reversal errors with a persistence guard. AdamW 0.001, weight decay 0.0001, cosine decay, 28 sampled rounds, at most 4,096 windows per round and batch 256. These are not full-dataset passes.</p>
<p>Adaptation withholds the final observed hour inside each individual target X. The predictor sees only its real preceding history. The loss combines target-input forecasting, 0.5 source replay and a 0.001 relative source-weight anchor. Twelve sampled adaptation rounds use learning rate 0.0002, source global/key guards of 10% and source-model fallback. No target original forecasting Y tensor or cross-window label reconstruction is used.</p>
<p>The small local dynamics branch fits next-hour changes from normalized recent own-channel trajectories, multivariate levels/changes/means and calendar features. Source ridge regularization is selected on source validation. Target normal equations combine within-input loss, 0.5 source replay and an anchor to source coefficients. A per-variable source/simplex mixture of neural, local and persistence forecasts is calibrated using source validation and target input tails only. Mixture adaptation has a 10% source retention guard. The inference model remains multivariate, with a fixed model at every test origin.</p>
<p>The final peak-calibrated hybrid retains those LiteTCN/dynamics experts, the earlier adapted hybrid and persistence. A causal regime indicator uses only the last three observations and past 48-hour scale to distinguish ordinary motion, approaching peaks, approaching troughs and observed flat/reversal motion. Twenty convex weights per variable combine the five source/adapted/persistence forecasts across those four regimes. This is a hybrid, not a claim that LiteTCN alone achieves these results.</p>
<p>Its loss directly matches the displayed daily event metrics: point MSE + 2 times MSE at the actual episode maximum/minimum + 0.1 ramp MSE. Calibration simulates 24 successive next-hour predictions entirely inside each validation X; at each inner origin only its preceding prefix is visible, with left padding to the original lookback length. Source calibration uses source-only experts. Target calibration balances relative source and observed-target episode losses with a 0.001 source-weight anchor, source point-error guard of 10%, target-input point-error guard of 5%, minimum 64 examples per regime, and source fallback. The first L validation origins are purged and episodes sampled every 25 origins to avoid repeated calibration labels. Actual episode extremes are loss labels during calibration, never regime features. No original target forecasting Y is accepted by the episode collector.</p>
<p>All seeds and model choices for each experiment were frozen before its test inference. The original dataset tests have been inspected during earlier project iterations, so these are development results, not independent confirmation. Historical cached training-period scalers remain unchanged for fair comparisons. The next-hour evaluation uses the same original benchmark origins, taking only their first future value; the final 23 otherwise-possible one-hour origins are not added.</p>
'''+''.join(content)+'''<h2>24-hour geometry-aware boosted adaptation</h2>
<p>This is a substantive alternative to a deeper TCN or a small residual MLP. A depth-4 CatBoost forecaster models the key temperature trajectory from observed history. Source controls compare 24 direct trajectory outputs against a geometry representation: 16 orthonormal DCT coefficients, separately weighted maximum/minimum temperature and circular peak/trough phase. Predicting expected peak height separately avoids treating the maximum of a conditional-mean curve as the expected maximum.</p>
<p>The geometric MultiRMSE target transform and high-span training weights emphasize extremes. Source validation and target-input validation choose raw reconstruction, partial range calibration or constrained phase/range calibration, and a 0/0.5/1 mixture with retained LiteTCN. Normalization uses observed-window statistics only. The geometry decoder can sharpen a forecast, but selection must also satisfy the ordinary source-MSE guard; sharper appearance alone is not accepted as evidence of accuracy.</p>
<p>Both controls use 400 source trees, 32 histogram borders, depth 4, leaf L2 regularization 10 and learning rate 0.06. Adaptation continues the source ensemble with up to 400 new trees at 0.04 using real L-H prefixes inside target X and 0.25 source replay. Source trees remain in the additive ensemble. The event selection score combines point MSE, errors at actual peaks/troughs, max/min-value error and ramp error. Adapter selection uses equal relative source-validation and target-input scores and a 5% source key-MSE guard. No actual future weather/load inputs or target future-test labels are supplied. CatBoost's multi-output objective and continuation APIs are documented in <a href="https://catboost.ai/docs/en/concepts/loss-functions-multiregression">the objective documentation</a> and <a href="https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit">the fit API</a>.</p>
'''+table(long)+''.join(long_content)+'''<h2>Third dataset: finance retained</h2>
<p>Finance remains the existing strictly input-only adapted model with a five-trading-day horizon. It was not retrained in this hourly-temperature follow-up. Compare it with baselines at its own horizon; do not compare its numerical MSE directly with different datasets.</p>
'''+table(finance)+'''<h2>Validation and efficiency</h2>
<p>'''+html.escape(count)+''' tests passed, covering input-tail/other-window isolation, real encoder learning, native tree serialization, exact forecast horizons, unchanged non-temperature forecasts in the 24-hour adapter, causal episode prefixes, sparse mixtures, CSV validation and the interface. The interface smoke suite checked '''+html.escape(ui_count)+''' dataset/family/domain combinations. A dedicated check prevents broadcasting a 24-hour legacy baseline against one-hour truth.</p>
<p>Artifact checks recompute selection, verify model/code/data hashes, check freeze timestamps and recompute complete source/target deployment test MSE. CPU latency is batch 1 after five warmups and 30 repeats. PyTorch uses one thread; the native tree predictor uses two threads when active. Neural parameter counts exclude tree leaves and ridge buffers; stored tensor counts and native leaf values are reported separately. Deployment size is the adapted forecaster only; auxiliary controls are reported separately.</p>
'''+table(efficiency)+'<details><summary>Artifact audit</summary><pre>'+html.escape(json.dumps(audit,indent=2))+'''</pre></details>
<h2>Reproduce and use</h2><pre>python -m src.nowcast_train
python -m src.nowcast_hybrid_train
python -m src.peak_nowcast_train
python -m src.event_boost_train
python -m src.event_boost_ablation
python -m src.check_horizon_models
python -m unittest discover -s tests -v
python -m src.forecast_horizons_report
streamlit run app/app.py</pre>
<p>Choose Peak-calibrated next-hour adaptation for the new peak-focused one-hour task, or the explicit 24-hour boosted option for a single-origin forecast. The first next-hour hybrid and earlier hybrid first-hour output remain selectable for the measured tradeoffs above. With a CSV containing historical actuals, the interface displays a rolling curve with hourly updated observations. With only past observations it outputs one next-hour forecast; a further rolling prediction requires a new observation. Source/adapted models, metrics and selection histories are saved under peak_next_hour, next_hour, next_hour_hybrid and event_boost in models/results.</p>
<p>Trained source/adapted models and experimental evidence are complete for these runs. The required official-template assessment report still needs the team's own verified narrative and actual contribution statements; these notes do not replace that deliverable.</p>'''
    path=ROOT/"report"/"IE643_Forecast_Horizons_Results.html"
    path.write_text('<!doctype html><html><head><meta charset="utf-8"><title>Forecast horizons and adaptation results</title><style>body{max-width:1140px;margin:36px auto;padding:0 24px;font:16px/1.55 system-ui;color:#17212b}h1,h2,h3{line-height:1.2}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}th,td{padding:8px;border-bottom:1px solid #d6dce3;text-align:left}img{width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:13px;color:#56616e}pre{background:#f2f5f8;padding:16px;overflow:auto}details{margin:16px 0}</style></head><body>'+document+'</body></html>',encoding="utf-8")
    print(path);print(rolling.to_string(index=False));print(long.to_string(index=False))


if __name__=="__main__":main()
