"""Build a standalone, quantitative report for strict label-free refinement."""
import base64
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DATASETS, MODEL_DIR, RESULTS_DIR, ROOT

OUT = RESULTS_DIR / "strict_refined"
FIG = OUT / "figures"


def table(frame):
    return frame.to_html(index=False, border=0, float_format=lambda v: f"{v:.5f}", escape=True)


def embedded(path, caption):
    return '<figure><img alt="' + html.escape(caption) + '" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '"><figcaption>' + html.escape(caption) + '</figcaption></figure>'


def previous(ds):
    initial = pd.read_csv(RESULTS_DIR / "improved" / ds / "metrics_all.csv")
    if ds != "energy":
        return initial[(initial.domain == "target") & (initial.method == "selected_adapted")]
    frame = pd.read_csv(RESULTS_DIR / "transfer_selection" / ds / "metrics_all.csv")
    selected = []
    for seed in (0, 1, 2):
        record = json.loads((RESULTS_DIR / "transfer_selection" / ds / f"seed_{seed}_selection.json").read_text())
        eligible = [n for n, v in record["scores"].items() if n.startswith("dlinear_") and v["eligible"]]
        chosen = min(eligible, key=lambda n: record["scores"][n]["proxy"]) if eligible else "dlinear_source"
        selected.append(frame[(frame.domain == "target") & (frame.seed == seed) & (frame.method == chosen)])
    return pd.concat(selected)


def main():
    FIG.mkdir(exist_ok=True)
    sections, headline, all_tables = [], [], []
    for ds, cfg in DATASETS.items():
        strict = pd.read_csv(OUT / ds / "metrics_all.csv")
        context = pd.read_csv(RESULTS_DIR / "strict_context" / ds / "metrics_all.csv")
        old = previous(ds)
        target = strict[strict.domain == "target"]
        current = target[target.method == "strict_adapted"]
        shorter = context[(context.domain == "target") & (context.method == "strict_adapted")]
        old_mse, new_mse, short_mse = old.MSE.mean(), current.MSE.mean(), shorter.MSE.mean()
        row = {"dataset": ds, "previous_best_MSE": old_mse, "strict_MSE": new_mse,
               "shorter_context_MSE": short_mse, "strict_gain_vs_previous_pct": 100 * (1 - new_mse / old_mse),
               "shorter_gain_vs_previous_pct": 100 * (1 - short_mse / old_mse),
               "previous_key_RMSE": old.key_RMSE.mean(), "strict_key_RMSE": current.key_RMSE.mean(),
               "shorter_context_key_RMSE": shorter.key_RMSE.mean(),
               "seasonal_naive_MSE": target[target.method == "seasonal_naive"].MSE.mean(),
               "input_mean_MSE": target[target.method == "input_mean"].MSE.mean()}
        headline.append(row)
        comparisons = [("Previous model (DLinear on energy; Patch MLP otherwise)", old),
                       ("Strict input-only ensemble", current), ("Shorter-context ensemble", shorter)]
        for method in ("strict_source", "strict_channel_source", "strict_channel_adapted", "seasonal_naive", "persistence", "input_mean"):
            comparisons.append((method, target[target.method == method]))
        records = [{"method": name, "MSE_mean": values.MSE.mean(), "MSE_seed_SD": values.MSE.std(),
                    "key_RMSE_mean": values.key_RMSE.mean(), "MAE_mean": values.MAE.mean(),
                    "MASE_train_mean": values.MASE_train.mean()} for name, values in comparisons]
        rows = pd.DataFrame(records)
        all_tables.append(rows.assign(dataset=ds))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        subset = rows.iloc[[0, 1, 2, 6, 7, 8]]
        for ax, metric, title in zip(axes, ("MSE_mean", "key_RMSE_mean"),
                                     ("Multivariate MSE (standardized)", f"{cfg['key']} RMSE ({cfg['key_unit']})")):
            ax.barh(np.arange(len(subset)), subset[metric], color=["#71859b", "#197c75", "#326fba", "#b7bfc8", "#b7bfc8", "#b7bfc8"])
            ax.set_yticks(np.arange(len(subset)), ["Previous", "Strict ensemble", "Shorter context", "Seasonal naive", "Persistence", "Input mean"])
            ax.invert_yaxis()
            ax.set_title(title, fontsize=10)
            ax.grid(axis="x", alpha=.2)
            ax.set_xlim(0, subset[metric].max() * 1.17)
            for i, value in enumerate(subset[metric]):
                ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)
        fig.suptitle(cfg["title"] + " · target development benchmark", fontsize=12)
        path = FIG / f"{ds}_strict_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        sections.append(f'<h2>{html.escape(cfg["title"])}</h2>' + table(rows) + embedded(path, "Means across three saved neural training seeds. The new ridge fits are deterministic; seed variation is not a confidence interval."))
        # Fixed examples ensure plots are not selected for attractive outcomes.
        a = np.load(RESULTS_DIR / "strict_context" / ds / "forecast_examples.npz")
        b = np.load(OUT / ds / "forecast_examples.npz")
        old_examples = np.load(RESULTS_DIR / "improved" / ds / "forecast_examples.npz")
        meta = json.loads((MODEL_DIR / "strict_refined" / ds / "meta.json").read_text())
        j = cfg["columns"].index(cfg["key"])
        mu, sd = meta["target_scaler"]["mean"][j], meta["target_scaler"]["std"][j]
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
        origins = [0, len(a["target_X"]) // 2, len(a["target_X"]) - 1]
        for ax, index in zip(axes, origins):
            steps = min(cfg["L"], 4 * cfg["H"])
            ax.plot(np.arange(-steps, 0), a["target_X"][index, -steps:, j] * sd + mu, color="#8b9097", label="Observed input")
            ax.plot(np.arange(cfg["H"]), a["target_Y"][index, :, j] * sd + mu, color="#171a20", lw=2, label="Actual future")
            for data, key, color, label in ((old_examples, "target_selected_adapted", "#bf863e", "Previous Patch MLP"),
                                           (b, "target_strict_adapted", "#197c75", "Strict ensemble"),
                                           (a, "target_strict_adapted", "#326fba", "Shorter context")):
                ax.plot(np.arange(cfg["H"]), data[key][index, :, j] * sd + mu, color=color, label=label)
            ax.axvline(-.5, color="#b8c0ca", ls=":")
            ax.set_ylabel(cfg["key_unit"], fontsize=9)
            ax.set_title(f"Fixed test-window index {a['target_indices'][index]}", fontsize=10)
            ax.grid(alpha=.15)
        axes[0].legend(ncol=3, fontsize=8)
        axes[-1].set_xlabel("Steps relative to forecast origin")
        path = FIG / f"{ds}_strict_forecasts.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        sections.append(embedded(path, "Seed-0 examples: first, middle, last of the same 256 evenly spaced test origins. Energy's earlier DLinear result is in the table; the plotted previous neural model is Patch MLP."))
        if ds == "finance":
            sections.append('<p>Finance forecasts remain smooth and often miss abrupt changes. Much of the gain comes from shrinking noisy estimates toward stable means; the improvement in average error leaves this qualitative limitation visible.</p>')
        source_rows = strict.groupby(["domain", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index()
        sections.append('<details><summary>Source retention and full domain tables</summary>' + table(source_rows) + '</details>')
    headline = pd.DataFrame(headline)
    headline.to_csv(OUT / "headline.csv", index=False)
    pd.concat(all_tables).to_csv(OUT / "comparison.csv", index=False)
    fresh = pd.read_csv(OUT / "confirmatory_2025" / "metrics_all.csv")
    fresh_table = fresh.groupby(["domain", "method"])[["MSE", "MAE", "key_RMSE", "MASE_train"]].mean().reset_index()
    fresh_table.to_csv(OUT / "confirmatory_2025" / "summary.csv", index=False)
    efficiency = pd.read_csv(OUT / "efficiency.csv")
    audit = json.loads((OUT / "artifact_audit.json").read_text())
    source_inventory = pd.DataFrame([{"dataset": ds, "source": c["source_label"], "target": c["target_label"],
                                      "variables": len(c["columns"]), "lookback": c["L"], "horizon": c["H"],
                                      "linear_short_context": c["L"] - c["H"]} for ds, c in DATASETS.items()])
    body = '''<h1>IE643: strict label-free improvement stage</h1>
<p>19 October 2026. Source models and adapted models are saved for all three benchmarks. Target forecasting labels are excluded from training and selection. The new ensemble adds channel-specific regularized forecast heads to the existing compact neural models.</p>
<h2>What improved, and what did not</h2>
<p>Climate and finance improve on their previously reported models. Finance now beats the stronger input-mean baseline. Energy oil-temperature RMSE improves, but its overall multivariate MSE still trails the earlier DLinear family slightly. Keep DLinear available when total energy MSE is the priority. No claim is made that every variable or every forecast improves.</p>
''' + table(headline) + '''<h2>Datasets and causal preparation</h2>''' + table(source_inventory) + '''
<p>The original processed CSVs and chronological splits are unchanged. Climate uses Open-Meteo hourly 2021–2024; energy uses ETTh1/ETTh2 hourly; finance uses five NSE IT and five pharma stocks, with four stationary features. Windows never cross stock boundaries. Scaling uses each domain's existing training-period statistics, frozen for validation/test and CSV inference. The original scaler convention uses the complete unlabeled target training series, including the last H points; new weight fitting and model selection do not consume target Y tensors.</p>
<h2>Stage 1: variable-specific source models</h2>
<p>For each variable, fit a small ridge forecasting head to its own recent lags, recent averages of all variables, and its own level and volatility. Source validation chooses normalization (none, mean-centered, last-centered, or window standardization) and ridge strength from 10^-5 through 1. This allows stationary returns to shrink toward a stable mean while retaining different dynamics for temperature, load and cloud cover. The original deeper-TCN comparison is documented in the earlier stage report; these are project-specific hybrid models, not named-paper reproductions.</p>
<h2>Stage 2: strictly input-only adaptation</h2>
<p>Each target X window is processed independently. Hide its final H observed values and forecast them from its own prefix. Do not stitch overlapping inputs into labeled target forecasting examples. The first strict model uses constant left padding for the same pretext as the earlier experiments. The shorter-context follow-up trains the linear branch with K=L-H recent values, so its within-input pretext uses K real observations and needs no padding. Neural branch proxies retain the earlier padding.</p>
<pre>L_adapter = MSE(withheld values inside target X)
          + replay × MSE(source forecast)
          + anchor × ||W - W_source||²
replay ∈ {0, .25, 1}; anchor ∈ {.0001, .001, .01, .1, 1, 10}</pre>
<p>The linear objective is solved exactly with regularized normal equations in standardized forecast units. Source normalization is retained. A source-validation + target-input-pretext score selects each variable's adapter; source forecasting MSE may increase by at most 10% on validation. The unadapted source is always a fallback. Existing Patch MLP and DLinear source/adapted checkpoints remain available as ensemble components.</p>
<h2>Stage 3: guarded ensemble selection</h2>
<pre>minimize 0.5 × source_val_MSE / source_reference_MSE
       + 0.5 × target_input_pretext_MSE / input_reference_MSE
       + .001 × ||mixture - source_mixture||²
subject to: mixture ≥ 0, sum(mixture) = 1,
            source_val_MSE ≤ 1.1 × source_reference_MSE</pre>
<p>Fit a separate convex mixture for each variable. Components are channel-specific source/adapted heads, Patch MLP source/adapted, DLinear source/adapted, input mean and training mean. Source mixture calibration uses source validation labels only. Target calibration receives target X only. Three previously trained neural seeds are reused; the new linear fits are deterministic. Deployment uses seed 0, not a test-picked seed. All three seeds are frozen and saved before each experiment's test evaluation.</p>
<h2>Evaluation protocol and metrics</h2>
<p>The original test periods have already influenced development and are explicitly reused development benchmarks. Model weights, mixture weights and hyperparameters are selected without their forecasting labels. Report standardized multivariate MSE/MAE, physical key-variable RMSE and MASE with fixed per-variable training denominators. Seed SD measures seed variability; overlapping forecast windows do not form independent samples. Report mean skill does not establish statistical significance or guarantee future improvements.</p>
''' + ''.join(sections) + '''<h2>Additional 2025 climate evaluation</h2>
<p>The first strict ensemble was frozen, hashed and saved before downloading 2025 observations. No training, tuning or model selection used this new period. Both domains have 8,737 rolling 24-hour forecasts, starting 1 January 2025. Historical December context is supplied for the first forecast. The seven-day overlap with cached 2024 data is identical for every variable in both domains, and API units match. The shorter-context follow-up is not claimed as independently confirmed on this period.</p>
''' + table(fresh_table) + '''
<p>The default <a href="https://open-meteo.com/en/docs/historical-weather-api">Open-Meteo historical API</a> best-match request is retained, with raw responses, parameters, units and hashes cached under data/confirmatory_2025/climate. This is not an ERA5-only dataset. New climate-period evidence is available; energy and finance still need an independent future period or domain.</p>
<h2>Efficiency and implementation tradeoff</h2>
''' + table(efficiency) + '''
<p>CPU timings use one thread, five warm-ups and 30 repeats; stored coefficients include learned linear buffers. The ensemble stores more coefficients and performs more computation than an individual component. It remains small in absolute terms, but it does not improve latency on every dataset. Branches with no meaningful mixture weight are skipped at inference.</p>
<h2>Interface and reproduction</h2>
<p>Run Streamlit, choose Strict label-free ensemble or Strict shorter-context ensemble, select Domain 1/2 and upload a chronological CSV. The existing date/value validation, multivariate plots, actual-vs-forecast comparison, forecast download and baseline metrics remain available. The first strict model is the default; previous models remain selectable. The benchmark tab includes both strict experiments and their selection records. New trained models are under models/strict_refined and models/strict_context.</p>
<pre>python -m src.strict_refinement --datasets climate energy finance --seeds 0 1 2 --threads 4
python -m src.strict_context --datasets climate energy finance --seeds 0 1 2 --threads 4
python -m src.confirm_climate
python -m src.check_strict
python -m src.strict_report
python -m unittest discover -s tests -v
streamlit run app/app.py</pre>
<p>These runners require the earlier saved improved and transfer-selection checkpoints. The earlier README commands reproduce those inputs. Versions are pinned in requirements-experiments.txt. Data/code hashes, selected coefficients, source guards, selection freeze records, full per-seed/domain metrics and deterministic qualitative examples are retained beside each experiment.</p>
<h2>Excluded broader-history pilot</h2>
<p>The pilot under models/refined and results/refined reconstructed complete forecasting pairs from overlapping target inputs. It produced preliminary numbers, but was excluded after the user required strict label-free adaptation. Its neural extension was stopped. Neither those results nor the incomplete history_refined run feed the strict models, interface, headline tables or 2025 evaluation. Retained artifacts are marked EXCLUDED_FROM_STRICT_RESULTS.</p>
<h2>Verification and remaining team steps</h2>
<pre>''' + html.escape(json.dumps(audit, indent=2)) + '''</pre>
<p>All 15 regression tests passed, including interface checks across 36 experiment/domain/dataset combinations. Checks cover withholding isolation, no cross-window influence, source-retention constraints, correct normal-equation objectives, checkpoint persistence and CSV validation. The running Streamlit health endpoint returns HTTP 200. Dataset approval from TAs and final team/report identity details remain team actions. The earlier extensive stage report contains the original data-extraction procedure, depth sweep and earlier experiments.</p>'''
    css = 'body{font:16px/1.55 system-ui,sans-serif;max-width:1240px;margin:40px auto;padding:0 24px;color:#162333}h1,h2{line-height:1.2}h2{margin-top:36px}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:13px;width:100%}td,th{padding:8px 10px;text-align:left;border-bottom:1px solid #dae0e7}th{background:#edf2f6}figure{margin:22px 0}img{max-width:100%;height:auto}figcaption{color:#526174;font-size:13px}pre{background:#edf2f6;padding:16px;overflow-x:auto;font-size:13px}details{margin:20px 0}'
    destination = ROOT / "report" / "IE643_Strict_Refinement_Report.html"
    destination.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>IE643 Strict Refinement</title><style>' + css + '</style><body>' + body + '</body></html>', encoding="utf-8")
    print(destination)
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
