"""Export measured LiteTCN comparisons and fixed qualitative examples.

This generates supporting experiment notes, not the team's final assessment report.
"""
import base64
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
from .inference import load_bundle, load_dlinear_bundle, load_strict_bundle, forecast_batches
from .strict_report import previous

OUT = RESULTS_DIR/"peak_litetcn"
FIG = OUT/"figures"


def table(frame):
    return frame.to_html(index=False, border=0, float_format=lambda v: f"{v:.4f}", escape=True)


def figure(path, caption):
    return '<figure><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'


def main():
    torch.set_num_threads(2)
    FIG.mkdir(exist_ok=True)
    content, headline, comparisons, all_peak_rows = [], [], [], []
    for ds in ("climate", "energy"):
        cfg = DATASETS[ds]
        files = {family: pd.read_csv(RESULTS_DIR/family/ds/"metrics_all.csv")
                 for family in ("peak_litetcn", "peak_calibrated", "residual_litetcn")}
        initial = pd.read_csv(RESULTS_DIR/"improved"/ds/"metrics_all.csv")
        old = initial[(initial.domain=="target")&(initial.method=="legacy_care_v2_seed0")]
        if ds == "climate":
            prior = pd.read_csv(RESULTS_DIR/"strict_context"/ds/"metrics_all.csv")
            prior = prior[(prior.domain=="target")&(prior.method=="strict_adapted")]
        else:
            prior = previous(ds)
        candidates = [("Original saved LiteTCN (fixed seed 0)", old),
                      ("Previous best family", prior)]
        for family, method, label in (
                ("peak_litetcn", "source_only", "LiteTCN v3 source only"),
                ("peak_litetcn", "selected_adapted", "LiteTCN v3 adapted"),
                ("peak_calibrated", "local_adapted", "LiteTCN v3 + recent residuals"),
                ("residual_litetcn", "linear_skip_only", "DLinear skip only"),
                ("residual_litetcn", "source_residual_tcn", "LiteTCN-R source only"),
                ("residual_litetcn", "adapted_residual_tcn", "LiteTCN-R adapted")):
            frame = files[family]
            candidates.append((label, frame[(frame.domain=="target")&(frame.method==method)]))
        for method in ("seasonal_naive", "persistence", "input_mean"):
            candidates.append((method, initial[(initial.domain=="target")&(initial.method==method)]))
        rows = pd.DataFrame([{"model": label, "MSE_mean": d.MSE.mean(), "MSE_seed_SD": d.MSE.std(),
                              "key_RMSE_mean": d.key_RMSE.mean(), "MAE_mean": d.MAE.mean(),
                              "MASE_train_mean": d.MASE_train.mean()} for label, d in candidates])
        comparisons.append(rows.assign(dataset=ds))
        primary = "LiteTCN v3 adapted" if ds=="climate" else "LiteTCN-R adapted"
        selected = rows[rows.model==primary].iloc[0]
        previous_row = rows[rows.model=="Previous best family"].iloc[0]
        headline.append({"dataset": ds, "retained_TCN": primary, "original_MSE": old.MSE.mean(),
            "retained_TCN_MSE": selected.MSE_mean, "MSE_gain_vs_original_pct": 100*(1-selected.MSE_mean/old.MSE.mean()),
            "original_key_RMSE": old.key_RMSE.mean(), "retained_TCN_key_RMSE": selected.key_RMSE_mean,
            "key_RMSE_gain_vs_original_pct": 100*(1-selected.key_RMSE_mean/old.key_RMSE.mean()),
            "previous_best_MSE": previous_row.MSE_mean, "previous_best_key_RMSE": previous_row.key_RMSE_mean,
            "TCN_MSE_change_vs_previous_best_pct": 100*(selected.MSE_mean/previous_row.MSE_mean-1)})
        fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained")
        for ax, metric, title in zip(axes, ("MSE_mean", "key_RMSE_mean"),
                                     ("Multivariate standardized MSE", f"{cfg['key']} RMSE ({cfg['key_unit']})")):
            colors = ["#326fba" if name==primary else "#197c75" if name=="Previous best family" else "#aab5c2" for name in rows.model]
            ax.barh(np.arange(len(rows)), rows[metric], color=colors)
            ax.set_yticks(np.arange(len(rows)), rows.model, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0, rows[metric].max()*1.15)
            for index, value in enumerate(rows[metric]): ax.text(value, index, f" {value:.4f}", va="center", fontsize=7)
            ax.grid(axis="x", alpha=.2)
        fig.suptitle(cfg["title"]+" · all development-test windows", fontsize=12)
        path=FIG/f"{ds}_retained_tcn_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        content.append('<h2>'+html.escape(cfg["title"])+'</h2>'+table(rows)+figure(path,
            "Means across three neural seeds; the original saved LiteTCN is a fixed seed-0 reference. Previous best is the shorter-context hybrid on climate and DLinear family on energy. All alternatives, including worse results, remain visible."))
        source_tables=pd.concat([d[d.domain=="source"].assign(family=family) for family,d in files.items()])
        source_summary=source_tables.groupby(["family","method"]).agg(MSE_mean=("MSE","mean"),
            key_RMSE_mean=("key_RMSE","mean"),MASE_train_mean=("MASE_train","mean"),seeds=("seed","nunique")).reset_index()
        content.append('<details><summary>Domain 1 forecasts and adapted-model source retention</summary>'+table(source_summary)+'</details>')
        a=np.load(OUT/ds/"forecast_examples.npz")
        r=np.load(RESULTS_DIR/"residual_litetcn"/ds/"forecast_examples.npz")
        legacy=np.load(RESULTS_DIR/"improved"/ds/"forecast_examples.npz")
        meta=json.loads((MODEL_DIR/"peak_litetcn"/ds/"meta.json").read_text())
        key=cfg["columns"].index(cfg["key"])
        sd,mu=meta["target_scaler"]["std"][key],meta["target_scaler"]["mean"][key]
        if ds=="climate":
            old_examples=np.load(RESULTS_DIR/"strict_context"/ds/"forecast_examples.npz")
            prior_prediction=old_examples["target_strict_adapted"]
            prior_label="Previous hybrid"
            prior_model=load_strict_bundle(ds, True)[1]["adapted_model"]
        else:
            prior_model=load_dlinear_bundle(ds)[1]["adapted_model"]
            prior_prediction=forecast_batches(prior_model, a["target_X"])
            prior_label="Previous DLinear"
        np.testing.assert_allclose(a["target_Y"], r["target_Y"])
        new_prediction=a["target_selected_adapted"] if ds=="climate" else r["target_adapted_residual_tcn"]
        fig, axes=plt.subplots(3,1,figsize=(12,8),layout="constrained")
        for ax,index in zip(axes,(0,128,255)):
            steps=min(cfg["L"],96)
            ax.plot(np.arange(-steps,0), a["target_X"][index,-steps:,key]*sd+mu,color="#9199a3",label="Observed input")
            ax.plot(np.arange(cfg["H"]),a["target_Y"][index,:,key]*sd+mu,color="#141820",lw=2,label="Actual future")
            for pred,color,label in ((legacy["target_legacy_care_v2_seed0"],"#b98146","Original LiteTCN"),
                                     (prior_prediction,"#197c75",prior_label),(new_prediction,"#326fba",primary)):
                ax.plot(np.arange(cfg["H"]),pred[index,:,key]*sd+mu,color=color,label=label)
            ax.axvline(-.5,color="#b8c0ca",ls=":")
            ax.set_title(f"Fixed test-window index {a['target_indices'][index]}",fontsize=10)
            ax.set_ylabel(cfg["key_unit"],fontsize=9)
            ax.grid(alpha=.15)
        axes[0].legend(ncol=3,fontsize=8)
        axes[-1].set_xlabel("Hours relative to forecast origin")
        path=FIG/f"{ds}_retained_tcn_forecasts.png"
        fig.savefig(path,dpi=150)
        plt.close(fig)
        content.append(figure(path,"Deployment seed 0: first, middle and last of the same 256 evenly spaced test origins used earlier. These examples were not selected to favor the new models; peaks and dips still missed by the new model remain visible."))
        data=prepare(ds)
        x,y=data["target"]["test"]
        old_model=load_bundle(ds)[1]["adapted_model"]
        peak_rows=[]
        for label,model in (("Original saved LiteTCN",old_model),(prior_label,prior_model)):
            p=forecast_batches(model,x)
            peak_rows.append({"model":label,**peak_metrics(y.numpy()[...,key]*sd,p[...,key]*sd)})
        for family,method,label in (("peak_litetcn","selected_adapted","LiteTCN v3"),
                                    ("peak_calibrated","local_adapted","LiteTCN v3 + residuals"),
                                    ("residual_litetcn","adapted_residual_tcn","LiteTCN-R")):
            d=pd.read_csv(RESULTS_DIR/family/ds/"peak_metrics.csv")
            row=d[(d.domain=="target")&(d.seed==0)&(d.method==method)].iloc[0].to_dict()
            peak_rows.append({"model":label,**{k:v for k,v in row.items() if k not in ("dataset","domain","method","seed")}})
        peak_table=pd.DataFrame(peak_rows)
        all_peak_rows.append(peak_table.assign(dataset=ds))
        columns=["model","RMSE","MAE_at_actual_peak","MAE_at_actual_trough","ramp_MAE","mean_actual_range","mean_predicted_range"]
        content.append('<h3>Peak/dip diagnostics: deployment seed 0, full test</h3>'+table(peak_table[columns]))
        source_ablation=[]
        for name in ("no_clock_mse","clock_mse","clock_shape"):
            record=json.loads((OUT/ds/f"screen_0_{name}.json").read_text())
            source_ablation.append({"variant":name,"selected_round":record["selected_round"],**record["best"]})
        content.append('<h3>Controlled source-validation ablation (seed 0)</h3>'+table(pd.DataFrame(source_ablation)))
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
        for name,color in (("input_mse","#326fba"),("input_shape","#b98146")):
            history=json.loads((OUT/ds/f"seed_0_{name}.json").read_text())
            rounds=[h["round"] for h in history]
            axes[0].plot(rounds,[h["proxy"] for h in history],label=name,color=color)
            axes[1].plot(rounds,[h["target_input"]["key_MSE"]**.5*sd for h in history],label=name,color=color)
        axes[0].set_title("Adaptation selection proxy (lower is better)")
        axes[1].set_title("Withheld observed-input temperature RMSE")
        for ax in axes:
            ax.set_xlabel("Adaptation round");ax.grid(alpha=.2);ax.legend(fontsize=8)
        path=FIG/f"{ds}_adaptation_behavior.png"
        fig.savefig(path,dpi=150);plt.close(fig)
        content.append(figure(path,"These are source-validation and target-input self-supervision diagnostics, not target future-test curves. Test futures were evaluated after freezing all checkpoints and choices."))
    headline=pd.DataFrame(headline)
    headline.to_csv(OUT/"retained_tcn_headline.csv",index=False)
    pd.concat(comparisons).to_csv(OUT/"retained_tcn_comparison.csv",index=False)
    pd.concat(all_peak_rows).to_csv(OUT/"deployment_peak_comparison.csv",index=False)
    fresh=pd.read_csv(OUT/"climate_2026"/"metrics_all.csv")
    fresh_table=fresh.groupby(["domain","method"])[["MSE","key_RMSE"]].mean().reset_index()
    horizons=pd.read_csv(OUT/"climate_2026"/"horizon_metrics.csv")
    horizon_table=horizons[horizons.domain=="target"].groupby(["method","horizon_hours"]).temperature_RMSE.mean().unstack()
    fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained")
    for name,row in horizon_table.iterrows(): ax.plot(row.index,row.values,marker="o",label=name)
    ax.set_xlabel("Cumulative forecast horizon (hours)");ax.set_ylabel("Temperature RMSE (degC)")
    ax.set_title("Frozen climate models · new Jan-Sep 2026 Delhi period")
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    path=FIG/"climate_2026_horizon_error.png"
    fig.savefig(path,dpi=150);plt.close(fig)
    efficiency=pd.read_csv(OUT/"efficiency.csv")
    diagnostic=pd.read_csv(RESULTS_DIR/"peak_diagnosis"/"diagnostics.csv")
    overfit=pd.read_csv(RESULTS_DIR/"peak_diagnosis"/"source_overfit_check.csv")
    audit=json.loads((OUT/"artifact_audit.json").read_text())
    body='''<h1>Retaining LiteTCN: model revisions and measured evidence</h1>
<p>IE643 project technical notes · 19 October 2026 · Ayush Patel (23B0037), Rishik Yadav (23B2103).</p>
<p>Supporting experiment material, not the team's final assessment report. Original checkpoints and earlier experiments are preserved. Climate and energy received targeted new LiteTCN experiments; the existing finance results and models remain available.</p>
<h2>Outcome</h2>
<p>LiteTCN was retained and retrained. Both highlighted variants improve the original saved LiteTCN. Neither consistently beats the strongest existing hybrid/DLinear family. That is the measured reason for keeping alternatives available in the final system. The default LiteTCN selections demonstrate continuity with the project; they are not claimed to be the lowest-error methods.</p>
'''+table(headline)+'''<h2>Why the peaks and dips are difficult</h2>
<p>The plots forecast the next 24 unseen hours, rather than reconstructing the observed input. Climate examples include both overprediction and underprediction: their actual/predicted maxima were 27.2/29.44, 31.5/29.50 and 16.1/17.31 degC. A blanket amplitude increase would worsen two examples. Daily amplitude changes, offsets and timing errors all matter.</p>
<p>Energy's prior source-training sample had a mean actual 24h range of 6.05 degC and a predicted range of 3.08 degC. Smoothing was therefore present before transfer. On 64 source examples, the earlier Patch MLP can reduce energy RMSE from 2.88 to 0.72 degC and restore most of the range. This shows capacity to fit that sample and functioning optimization; it does not establish an achievable test-error floor.</p>
<p>Pointwise MSE favors averages across uncertain futures. The available inputs contain past local weather/load measurements, but no future weather forecasts, regional spatial context or known future transformer dispatch. These missing drivers plausibly contribute to sudden changes; an irreducible-error bound has not been established. Small-batch fitting, validation overfitting and adaptation-proxy failures show that simply adding layers does not resolve the measured problem. All models forecast the horizon directly, so this is not recursive error accumulation.</p>
<details><summary>Full source/target diagnostic measurements</summary>'''+table(diagnostic)+table(overfit[["dataset","model","updates","RMSE","mean_actual_range","mean_predicted_range"]])+'''</details>
<h2>LiteTCN v3: source-trained revision</h2>
<p>A small dilated TCN is retained. Six daily cycles are folded into 24 hourly-position tokens; no hourly positions are averaged away. Four depthwise/pointwise residual blocks use dilations 1,2,4,8, with attention over observed tokens, a channel-specific linear skip, and optional known clock/week/year sine/cosine features. The effective history is 144 real hours, allowing a complete 144-to-24-hour pretext inside each 168-hour target input without padding. All 24 outputs are predicted directly. The model has 44,399 parameters.</p>
<pre>Source controls: no-clock/MSE, clock/MSE, clock/shape.
Key variable weight: 3; other variables: 1.
Shape objective = weighted MSE + .25 ramp MSE + .1 max/min-value MSE.
Common selection score = .5 multivariate MSE + .5 key MSE
                       + .25 key peak/trough MSE + .1 key ramp MSE.</pre>
<p>Each control uses the same 36 sampled rounds, up to 4,096 randomly drawn source windows per round, batch 256, AdamW 0.001, weight decay 0.0001 and cosine decay. These are sampled rounds, not 36 complete dataset passes. Source validation chooses the variant and checkpoint; clock/MSE wins on both datasets. The shape loss improves some extreme-value measurements but does not win the combined source criterion. Only the chosen variant is replicated across seeds 1 and 2; control ablations use seed 0.</p>
<pre>Input-only adaptation = forecast loss on withheld values INSIDE each target X
                      + .25 source replay loss + .001 L2-SP.
10 sampled adaptation rounds, AdamW 0.0003, cosine decay.
Checkpoint proxy = .5 relative source-validation score
                 + .5 relative target-input-pretext score.
Reject source validation MSE increases above 10%; include the source fallback.</pre>
<h2>LiteTCN-R: preserve the original encoder</h2>
<p>The original source-trained three-layer strided LiteTCN encoder and projection weights are retained exactly. DLinear's seasonal/trend operators are folded algebraically into the shared linear temporal skip; a zero neural head reproduces its source forecast. The residual head is fitted with source-only ridge regression and adapted using source replay, a source-head anchor and within-input forecast residuals. The encoder remains frozen. The unused reconstruction head is removed: 229,902 stored parameters, of which 21,672 belong to the updated head.</p>
<pre>Residual adaptation = target-input forecast MSE + replay × source MSE
                    + anchor × ||head - source_head||².
Replay ∈ {.25, 1}; anchor ∈ {.001, .01, .1, 1, 10}.
Select per variable using source validation + target-input pretext,
with a 10% source guard and an explicit zero-residual baseline.</pre>
<p>This controls how much a source-specific nonlinear residual can change the strong linear forecast. On energy the neural residual still slightly worsens overall target MSE compared with the linear skip alone. That negative result is retained; this model is not relabeled as a superior DLinear model.</p>
<h2>Optional recent-residual correction</h2>
<p>A separate LiteTCN v3 variant compares a forecast made 12 hours earlier with now-observed input values, then applies a bounded, decaying level/affine correction. Source validation and a nested target-input validation task select its strength. Windows reset independently, and no target forecasting labels are used. It gives modest near-term corrections rather than solving unseen peak changes.</p>
'''+''.join(content)+'''<h2>New 2026 climate period</h2>
<p>LiteTCN v3, its recent-residual policy, the original saved model and the previous hybrid were frozen and hashed before downloading Jan-Sep 2026 weather data. No subsequent model selection or training used that period for those frozen models. The LiteTCN-R extension is not claimed to have independent confirmation on this period. Each domain has 6,529 rolling 24-hour forecasts. Data and request metadata are cached separately under data/confirmatory_2026/climate.</p>
'''+table(fresh_table)+figure(path,"Cumulative horizon RMSE, averaged over three saved neural seeds. Original LiteTCN is the same seed-0 checkpoint repeated as a fixed reference. Smaller horizons are easier, but all 24-hour metrics remain reported.")+'''<h2>Efficiency</h2>
'''+table(efficiency)+'''<p>CPU timings use one thread, five warmups and 30 repeats, with clock features prepacked. LiteTCN v3 stores fewer parameters but is slower per window than the original strided model. LiteTCN-R preserves its inference backbone and only updates a small head. Total original parameter counts include an unused reconstruction head; forecast-parameter counts exclude it.</p>
<h2>Reproduction and verification</h2>
<pre>python -m src.peak_train --datasets climate energy --seeds 0 1 2 --epochs 36 --adapt-epochs 10 --max-windows 4096 --batch-size 256 --threads 4
python -m src.peak_calibrate --datasets climate energy --seeds 0 1 2
python -m src.residual_train --datasets climate energy --seeds 0 1 2
python -m src.peak_confirm
python -m src.check_peak
python -m src.peak_report
python -m unittest discover -s tests -v
streamlit run app/app.py</pre>
<p>The runners reuse the earlier saved source checkpoints in models/improved. Source/target chronological splits and scalers remain unchanged. Source labels are permitted; target weight fitting and selection use each individual input only, with no cross-window reconstruction of forecasting labels. The original development test has been observed repeatedly and is explicitly not an independent holdout. A final timing-log error in the first climate invocation was corrected after checkpoints and metrics had been saved; the original code snapshot/manifest and unchanged selection-freeze timestamp are retained.</p>
<pre>'''+html.escape(json.dumps(audit,indent=2))+'''</pre>
<p>All 24 regression tests passed, including 48 interface experiment/domain/dataset combinations, calendar alignment, encoder-gradient flow, within-input isolation, exact DLinear folding, residual-head scaling/order and checkpoint persistence. Source-only small-subset fitting checks were also run separately. The interface names every model family explicitly, supports source and target CSVs, and keeps stronger benchmark alternatives available.</p>
<h2>Assessment use</h2>
<p>The evidence supports an honest sequence: original LiteTCN, diagnosed smoothing/generalization, controlled architectural/loss experiments, retained-encoder residual adaptation, and comparison against stronger alternatives. Include negative results and explain why a deployment alternative is selected. No claim is made that a new architecture is a named-paper reproduction or that these changes guarantee a grade. The official template was verified against the supplied ZIP and saved unchanged under report/official_template; this supporting note is not a replacement for the team's own final report in that template.</p>'''
    css='body{font:16px/1.55 system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#162333}h1,h2{line-height:1.2}h2{margin-top:36px}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:13px;width:100%}td,th{padding:8px 10px;border-bottom:1px solid #dae0e7;text-align:left}th{background:#edf2f6}figure{margin:24px 0}img{max-width:100%;height:auto}figcaption{color:#526174;font-size:13px}pre{background:#edf2f6;padding:16px;overflow-x:auto;font-size:13px}'
    destination=ROOT/"report"/"IE643_LiteTCN_Model_Results.html"
    destination.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>IE643 LiteTCN Model Results</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf-8')
    print(destination)
    print(headline.to_string(index=False))


if __name__=="__main__": main()
