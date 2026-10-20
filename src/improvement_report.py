"""Build plots and a standalone HTML stage report from completed improvement runs."""
import html
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DATASETS, MODEL_DIR, PROC_DIR, RESULTS_DIR, ROOT
from .inference import load_dlinear_bundle, forecast_batches

OUT = RESULTS_DIR / "improved"
FIG = OUT / "figures"
NAMES = {"selected_source": "Validation-selected source", "selected_adapted": "Selected adaptation",
         "legacy_care_v2_seed0": "Original CARE-DA v2 (seed 0)", "legacy_source_seed0": "Original source (seed 0)",
         "supervised_reference": "Supervised target reference", "seasonal_naive": "Seasonal naive",
         "persistence": "Persistence", "input_mean": "Input mean", "tcn3": "LiteTCN 3 layers",
         "tcn5": "LiteTCN 5 layers", "tcn7": "LiteTCN 7 layers", "patch_mlp": "Patch MLP", "dlinear": "RevIN DLinear",
         "replay_inner": "Replay + inner forecast", "replay_inner_mmd": "Replay + inner forecast + MMD"}
NAMES.update({"main_source": "Initial selected source", "main_replay_inner": "Initial model + inner",
              "main_replay_inner_mmd": "Initial model + inner + MMD", "dlinear_source": "DLinear source",
              "dlinear_replay_inner": "DLinear + inner", "dlinear_replay_inner_mmd": "DLinear + inner + MMD",
              "history_selected_source": "History-selected source", "history_selected_adapted": "History-selected adaptation"})


def fmt_table(df):
    return df.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}", na_rep="—", escape=True)


def figure_tag(path, caption):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def plot_dataset(ds, df):
    cfg = DATASETS[ds]
    target = df[df.domain == "target"]
    order = ["seasonal_naive", "persistence", "input_mean", "legacy_care_v2_seed0",
             "tcn3", "tcn5", "tcn7", "patch_mlp", "dlinear", "selected_source", "selected_adapted", "supervised_reference"]
    stats = target.groupby("method")[["MSE", "key_RMSE"]].agg(["mean", "std"]).reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
    colors = ["#168075" if n == "selected_adapted" else "#415c80" if n == "selected_source" else "#a5b4c5" for n in order]
    for ax, metric, title in zip(axes, ["MSE", "key_RMSE"], ["Multivariate MSE (standardized)", f"{cfg['key']} RMSE ({cfg['key_unit']})"]):
        ax.barh(np.arange(len(order)), stats[(metric, "mean")], xerr=stats[(metric, "std")].fillna(0),
                color=colors, capsize=2)
        ax.set_yticks(np.arange(len(order)), [NAMES[n] for n in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle(f"{ds.title()} target test · mean ± sample SD across {target.seed.nunique()} seeds", fontsize=13)
    path = FIG / f"{ds}_benchmark.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths = [(path, "Full target test metrics. Error bars reflect training-seed variation, not statistical confidence intervals. Historical seed-0 references are fixed.")]

    examples = OUT / ds / "forecast_examples.npz"
    if examples.exists():
        a = np.load(examples)
        meta = json.loads((MODEL_DIR / "improved" / ds / "meta.json").read_text())
        key = cfg["columns"].index(cfg["key"])
        mu, sd = meta["target_scaler"]["mean"][key], meta["target_scaler"]["std"][key]
        X, Y = a["target_X"], a["target_Y"]
        origins = [0, len(X) // 2, len(X) - 1]
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), layout="constrained")
        L, H = cfg["L"], cfg["H"]
        for ax, i in zip(axes, origins):
            steps = min(L, 4 * H)
            ax.plot(np.arange(-steps, 0), X[i, -steps:, key] * sd + mu, color="#818b97", label="Observed input")
            ax.plot(np.arange(H), Y[i, :, key] * sd + mu, color="#151a20", lw=2, label="Actual future")
            for method, color, style in [("legacy_care_v2_seed0", "#df8f24", "--"),
                                          ("selected_source", "#826ab2", ":"),
                                          ("selected_adapted", "#168075", "-")]:
                ax.plot(np.arange(H), a[f"target_{method}"][i, :, key] * sd + mu,
                        color=color, ls=style, label=NAMES[method])
            ax.axvline(-.5, color="#b5bdc5", ls=":")
            ax.set_title(f"Deterministic example {i + 1} · test-window index {a['target_indices'][i]}", fontsize=10)
            ax.set_ylabel(cfg["key_unit"])
        axes[0].legend(ncol=3, fontsize=8, loc="best")
        axes[-1].set_xlabel("Steps relative to forecast origin")
        fig.suptitle(f"{ds.title()} · {cfg['key']} · deployment seed {meta['training_seed']}")
        path = FIG / f"{ds}_forecast_examples.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append((path, "First, middle and last stored examples, fixed without inspecting errors. Curves use the deployment seed; metrics use every test window."))

    history = OUT / ds / "seed_0_replay_inner.json"
    if history.exists():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), layout="constrained")
        for name in ("replay_inner", "replay_inner_mmd"):
            h = pd.DataFrame(json.loads((OUT / ds / f"seed_0_{name}.json").read_text()))
            for ax, metric in zip(axes, ["source_val_mse", "target_input_mse", "selection_score"]):
                ax.plot(h.epoch, h[metric], marker="o", ms=3, label=NAMES[name])
                ax.set_title(metric.replace("_", " "), fontsize=10)
                ax.set_xlabel("Adaptation epoch")
                ax.grid(alpha=.2)
        axes[-1].axhline(1, color="gray", ls=":")
        axes[0].legend(fontsize=7)
        path = FIG / f"{ds}_adaptation.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append((path, "Adaptation logs for seed 0. These curves use source validation futures and target validation inputs; target test errors are excluded from checkpoint selection."))
    return paths


def plot_transfer(ds, df):
    cfg = DATASETS[ds]
    order = ["legacy_care_v2_seed0", "main_source", "main_replay_inner", "main_replay_inner_mmd",
             "dlinear_source", "dlinear_replay_inner", "dlinear_replay_inner_mmd", "history_selected_adapted"]
    target = df[df.domain == "target"]
    grouped = target.groupby("method")[["MSE", "key_RMSE"]].agg(["mean", "std"]).reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, metric in zip(axes, ("MSE", "key_RMSE")):
        ax.barh(np.arange(len(order)), grouped[(metric, "mean")],
                xerr=grouped[(metric, "std")].fillna(0), capsize=2,
                color=["#168075" if name == "history_selected_adapted" else "#9cacbf" for name in order])
        ax.set_yticks(np.arange(len(order)), [NAMES[n] for n in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Standardized MSE" if metric == "MSE" else f"{cfg['key']} RMSE ({cfg['key_unit']})", fontsize=10)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle(f"{ds.title()} · exploratory transfer selection · {target.seed.nunique()} seeds")
    path = FIG / f"{ds}_transfer_benchmark.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures = [(path, "Follow-up experiment on the reused development test period. Source and adapted DLinear variants remain visible alongside the proxy-selected result.")]
    a = np.load(RESULTS_DIR / "transfer_selection" / ds / "forecast_examples.npz")
    meta = json.loads((MODEL_DIR / "transfer_selection" / ds / "meta.json").read_text())
    key = cfg["columns"].index(cfg["key"])
    mu, sd = meta["target_scaler"]["mean"][key], meta["target_scaler"]["std"][key]
    X, Y = a["target_X"], a["target_Y"]
    L, H = cfg["L"], cfg["H"]
    _, dlinear_models = load_dlinear_bundle(ds)
    dlinear_forecasts = forecast_batches(dlinear_models["adapted_model"], X)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), layout="constrained")
    for ax, i in zip(axes, (0, len(X) // 2, len(X) - 1)):
        steps = min(L, 4 * H)
        ax.plot(np.arange(-steps, 0), X[i, -steps:, key] * sd + mu, color="#818b97", label="Observed input")
        ax.plot(np.arange(H), Y[i, :, key] * sd + mu, color="#151a20", lw=2, label="Actual future")
        for method, color, style in (("legacy_care_v2_seed0", "#df8f24", "--"),
                                      ("history_selected_source", "#826ab2", ":"),
                                      ("history_selected_adapted", "#168075", "-")):
            ax.plot(np.arange(H), a[f"target_{method}"][i, :, key] * sd + mu, color=color, ls=style, label=NAMES[method])
        ax.plot(np.arange(H), dlinear_forecasts[i, :, key] * sd + mu, color="#2a69ad", label="DLinear family adaptation")
        ax.axvline(-.5, color="#b5bdc5", ls=":")
        ax.set_ylabel(cfg["key_unit"])
        ax.set_title(f"Fixed example {i + 1} · test-window index {a['target_indices'][i]}", fontsize=10)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Steps relative to forecast origin")
    fig.suptitle(f"{ds.title()} · selected using target history · deployment seed {meta['training_seed']}")
    path = FIG / f"{ds}_transfer_forecasts.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append((path, "First, middle and last stored origins for the target-history-selected deployment. All test windows, rather than just these examples, contribute to reported metrics."))
    return figures


def build():
    FIG.mkdir(parents=True, exist_ok=True)
    sections, all_metrics, headline = [], [], []
    transfer_sections, transfer_headline = [], []
    for ds, cfg in DATASETS.items():
        completed = sorted((OUT / ds).glob("seed_[0-9]*.csv"))
        if not completed:
            continue
        df = pd.concat([pd.read_csv(p) for p in completed], ignore_index=True)
        all_metrics.append(df)
        target = df[df.domain == "target"]
        grouped = target.groupby("method")[["MSE", "MAE", "MASE_train", "key_RMSE", "params"]].agg(["mean", "std"])
        summary = grouped.copy()
        summary.columns = ["_".join(c) for c in summary.columns]
        summary = summary.reset_index()
        old = float(grouped.loc["legacy_care_v2_seed0", ("MSE", "mean")])
        new = float(grouped.loc["selected_adapted", ("MSE", "mean")])
        source_mse = float(grouped.loc["selected_source", ("MSE", "mean")])
        key_old = float(grouped.loc["legacy_care_v2_seed0", ("key_RMSE", "mean")])
        key_new = float(grouped.loc["selected_adapted", ("key_RMSE", "mean")])
        depth_changes = {name: 100 * (float(grouped.loc[name, ("MSE", "mean")]) /
                                     float(grouped.loc["tcn3", ("MSE", "mean")]) - 1)
                         for name in ("tcn5", "tcn7")}
        baseline_name = min(("seasonal_naive", "persistence", "input_mean"),
                            key=lambda name: float(grouped.loc[name, ("MSE", "mean")]))
        baseline_error = float(grouped.loc[baseline_name, ("MSE", "mean")])
        baseline_change = 100 * (new / baseline_error - 1)
        headline.append({"Dataset": ds, "New seeds": target.seed.nunique(), "Original v2 MSE": old,
                         "Selected source MSE": source_mse, "Selected adapted MSE": new,
                         "MSE change vs original v2 (%)": 100 * (new / old - 1),
                         "Adaptation change vs new source (%)": 100 * (new / source_mse - 1),
                         "Original key RMSE": key_old, "New key RMSE": key_new})
        sel = [json.loads(p.read_text()) for p in sorted((OUT / ds).glob("seed_*_selection.json"))]
        selection_table = pd.DataFrame([{"Seed": s["seed"], "Source architecture": s["architecture"],
                                         "Adaptation recipe": s["adaptation"],
                                         "Selected epoch": s["epochs_selected"][s["adaptation"]],
                                         "Selection proxy": s["adaptation_proxy"][s["adaptation"]]} for s in sel])
        architecture_scores = pd.DataFrame([{"Seed": s["seed"], **s["source_val_mse"]} for s in sel])
        inventory = []
        for role in ("source", "target"):
            frame = pd.read_csv(PROC_DIR / ds / f"{cfg[role]}.csv")
            series = frame.groupby("series") if "series" in frame else [(cfg[role], frame)]
            for name, g in series:
                dates = pd.to_datetime(g.date).sort_values().reset_index(drop=True)
                n = len(dates)
                a, b = int(n * cfg["split"][0]), int(n * sum(cfg["split"][:2]))
                inventory.append({"Domain": role, "Series": name, "Rows": n,
                                  "Train through": str(dates.iloc[a - 1].date()),
                                  "Validation through": str(dates.iloc[b - 1].date()),
                                  "Test through": str(dates.iloc[-1].date()),
                                  "Missing cells": int(g[cfg["columns"]].isna().sum().sum())})
        content = [f"<h2>{html.escape(cfg['title'])}</h2>",
                   f"<p>Input length {cfg['L']}; forecast horizon {cfg['H']}; seasonal period {cfg['season']}. "
                   f"Variables: {html.escape(', '.join(cfg['columns']))}. Key variable: {html.escape(cfg['key'])}.</p>",
                   f"<p>Compared with the 3-layer LiteTCN, the 5-layer model changes target MSE by {depth_changes['tcn5']:+.2f}% and the 7-layer model by {depth_changes['tcn7']:+.2f}%. "
                   f"The selected adapted model changes MSE by {baseline_change:+.2f}% against the strongest no-training baseline, {NAMES[baseline_name]} (MSE {baseline_error:.4f}). "
                   f"{'The learned model does not beat this baseline, so claims of useful forecast skill must be limited.' if baseline_change >= 0 else 'The learned model improves on this baseline on this test period.'}</p>",
                   "<h3>Chronological data inventory</h3>", fmt_table(pd.DataFrame(inventory)),
                   "<h3>Validation decisions</h3>", fmt_table(selection_table),
                   "<p>Architecture selection minimizes source validation multivariate MSE. These scores explain why a deeper model was or was not selected.</p>",
                   fmt_table(architecture_scores), "<h3>Full target test results</h3>",
                   "<p>All multivariate metrics average variables equally after each domain is scaled with its own training statistics. Lower is better. "
                   "The supervised reference uses target future labels; all other adapted models use target input histories only.</p>",
                   fmt_table(summary[["method", "MSE_mean", "MSE_std", "MAE_mean", "MAE_std", "MASE_train_mean"]]),
                   fmt_table(summary[["method", "key_RMSE_mean", "key_RMSE_std", "params_mean"]])]
        for plot_path, caption in plot_dataset(ds, df):
            content.append(figure_tag(plot_path, caption))
        source_summary = df[df.domain == "source"].groupby("method")[["MSE", "key_RMSE"]].mean().reset_index()
        content.extend(["<h3>Source retention</h3>", fmt_table(source_summary)])
        sections.append("\n".join(content))
        transfer_folder = RESULTS_DIR / "transfer_selection" / ds
        transfer_completed = sorted(transfer_folder.glob("seed_[0-9]*.csv"))
        if transfer_completed:
            td = pd.concat([pd.read_csv(p) for p in transfer_completed], ignore_index=True)
            ts = td[td.domain == "target"].groupby("method")[["MSE", "MAE", "key_RMSE"]].agg(["mean", "std"])
            old_mse = float(ts.loc["legacy_care_v2_seed0", ("MSE", "mean")])
            final_mse = float(ts.loc["history_selected_adapted", ("MSE", "mean")])
            transfer_headline.append({"Dataset": ds, "Family": "Common proxy", "Seeds": td.seed.nunique(), "Original v2 MSE": old_mse,
                                      "History-selected MSE": final_mse, "MSE change (%)": 100 * (final_mse / old_mse - 1),
                                      "Original key RMSE": float(ts.loc["legacy_care_v2_seed0", ("key_RMSE", "mean")]),
                                      "History-selected key RMSE": float(ts.loc["history_selected_adapted", ("key_RMSE", "mean")])})
            decisions = []
            dlinear_scores = []
            for path in sorted(transfer_folder.glob("seed_*_selection.json")):
                decision = json.loads(path.read_text())
                if decision["seed"] in set(td.seed):
                    decisions.append({"Seed": decision["seed"], "Chosen candidate": decision["chosen"],
                                      "Architecture": decision["model_spec"]["architecture"],
                                      "Proxy": decision["scores"].get(decision["chosen"], {"proxy": 1})["proxy"]})
                    eligible_dlinear = [name for name, score in decision["scores"].items() if name.startswith("dlinear_") and score["eligible"]]
                    family_choice = min(eligible_dlinear, key=lambda name: decision["scores"][name]["proxy"]) if eligible_dlinear else "dlinear_source"
                    family_row = td[(td.seed == decision["seed"]) & (td.domain == "target") & (td.method == family_choice)].iloc[0]
                    dlinear_scores.append({"MSE": family_row.MSE, "key_RMSE": family_row.key_RMSE})
            family_mean = pd.DataFrame(dlinear_scores).mean()
            dlinear_source_mse = td[(td.domain == "target") & (td.method == "dlinear_source")].MSE.mean()
            family_adaptation_change = 100 * (family_mean.MSE / dlinear_source_mse - 1)
            transfer_headline.append({"Dataset": ds, "Family": "DLinear", "Seeds": len(dlinear_scores), "Original v2 MSE": old_mse,
                                      "History-selected MSE": family_mean.MSE, "MSE change (%)": 100 * (family_mean.MSE / old_mse - 1),
                                      "Original key RMSE": float(ts.loc["legacy_care_v2_seed0", ("key_RMSE", "mean")]),
                                      "History-selected key RMSE": family_mean.key_RMSE})
            ts.columns = ["_".join(c) for c in ts.columns]
            chunk = [f"<h3>{ds.title()} follow up</h3>",
                     f"<p>The proxy-selected DLinear family has target MSE {family_mean.MSE:.4f}, compared with {dlinear_source_mse:.4f} before adaptation ({family_adaptation_change:+.2f}%). "
                     "This separates the contribution of the architecture from that of adaptation. In particular, the energy gain comes mainly from the linear architecture; adapting it does not consistently reduce multivariate MSE.</p>",
                     fmt_table(pd.DataFrame(decisions)), fmt_table(ts.reset_index())]
            for plot, caption in plot_transfer(ds, td):
                chunk.append(figure_tag(plot, caption))
            transfer_sections.append("\n".join(chunk))
    if not all_metrics:
        raise RuntimeError("No completed runs. Run python -m src.improve first.")
    efficiency_path = OUT / "inference_latency.csv"
    efficiency = ""
    if efficiency_path.exists():
        metadata = json.loads((OUT / "inference_latency_meta.json").read_text())
        efficiency = ("<h3>Measured inference efficiency</h3><p>Warm CPU forward passes only; CSV loading, scaling, checkpoint loading and interface rendering are excluded. "
                      f"CPU threads: {metadata['threads']}; warm-up calls: {metadata['warmups']}; measured repeats: {metadata['repeats']}. "
                      f"Processor: {html.escape(metadata['processor'])}. Timing varies with hardware and background load. Checkpoint size includes all serialized parameters.</p>" +
                      fmt_table(pd.read_csv(efficiency_path)))
    pd.concat(all_metrics).to_csv(OUT / "metrics_all.csv", index=False)
    headline_df = pd.DataFrame(headline)
    headline_df.to_csv(OUT / "headline.csv", index=False)
    transfer_section = ""
    transfer_summary_top = ""
    if transfer_headline:
        transfer_headline_df = pd.DataFrame(transfer_headline)
        transfer_headline_df.to_csv(RESULTS_DIR / "transfer_selection" / "headline.csv", index=False)
        transfer_summary_top = ("<h2>Measured compact model alternatives</h2><p>Common proxy compares the compact candidates using source validation and observed target history. DLinear restricts that same proxy to the linear model family. "
                                "These are exploratory comparisons on the reused development test period; the lowest test error does not establish an independently validated deployment choice. Negative changes mean lower error.</p>" +
                                fmt_table(transfer_headline_df.drop(columns=["Original key RMSE", "History-selected key RMSE"])) +
                                fmt_table(transfer_headline_df[["Dataset", "Family", "Original key RMSE", "History-selected key RMSE"]]))
        transfer_section = """<h2>Follow up on architecture selection using target history</h2>
<p>The initial energy experiment exposed a limitation of choosing an architecture only on source validation: the DLinear source model transferred better than the source-validation winner. We therefore ran an exploratory extension on every benchmark and seed, adapting DLinear with the same two recipes and the same update budget. The initial source model and its two adapted variants are compared with DLinear and its two adapted variants.</p>
<pre>common proxy = 0.5 source_validation_MSE / legacy_source_validation_MSE
             + 0.5 target_input_pretext_MSE / legacy_source_input_pretext_MSE</pre>
<p>Unlike a proxy normalized separately for each architecture, this common reference makes the scores comparable across candidates. Candidates must keep source validation MSE within 10 percent of the legacy source model. If none pass, the legacy source is retained. The candidate with the smallest eligible proxy is saved before test evaluation. No target future tensor enters this selection. The corresponding source architecture is saved for Domain 1, and the chosen adapter for Domain 2.</p>
<p>This is explicitly a follow-up motivated by observed development test results, rather than a preregistered independent holdout result. A fresh time period or new domain is needed to validate its generalization. A validation proxy can still misrank transfer performance. The interface therefore exposes the initial source-validation selector, the common target-history selector, and an explicit DLinear family experiment. Within the DLinear family, the same input-only proxy chooses the source fallback or an adapter. The tables below retain both the common-proxy choice and the DLinear family outcome.</p>
""" + "\n".join(transfer_sections)
    body = """<h1>IE643 forecasting and domain adaptation experiment report</h1>
<p class="subtitle">Stage update · 19 October 2026</p>
<p>This report tests whether deeper LiteTCNs or alternative lightweight models improve the existing project. It records the data, fixed experiment design, architecture and adaptation decisions, full test metrics, limitations and interface methodology. Read the numerical changes below with both multivariate MSE and the key-variable RMSE; a gain in one does not guarantee a gain in the other.</p>
""" + transfer_summary_top + """<h2>Initial sweep with architecture chosen on source validation</h2>
""" + fmt_table(headline_df.drop(columns=["Original key RMSE", "New key RMSE"])) + fmt_table(headline_df[["Dataset", "Original key RMSE", "New key RMSE"]]) + """
<p>The existing CARE-DA v2 reference is the saved seed-0 deployment checkpoint. New models report means and sample standard deviations across the completed seeds. Repeating the historical checkpoint in each seed does not provide independent historical runs. Negative percentage changes indicate lower error. The source-to-adapted change isolates the contribution of adaptation from the contribution of the new architecture and training objective.</p>
<h2>Stage 1 Audit of the previous implementation</h2>
<p>The previous spectral term compared raw source and target input FFTs. Since these inputs do not depend on model parameters, this term had no training gradient. The implementation now computes spectra from the learned temporal encoder features, and a regression test verifies nonzero encoder gradients. The old checkpoints retain their historical behavior; changing the code does not retroactively change trained models. The new adaptation experiment below does not use the spectral term.</p>
<p>Old MASE values divide by each input window's average seasonal differences. Those values remain available for historical comparisons, but cannot be interpreted as a direct win against the future seasonal-naive forecast. The new MASE_train uses per-variable seasonal absolute differences from the chronological training segment as fixed denominators. The app labels its local version Window-scaled MAE. Gap closure is now undefined when the supervised reference fails to outperform the source baseline.</p>
<h2>Stage 2 Data and evaluation protocol</h2>
<p>The existing processed datasets are reused without downloading new observations. Climate uses Shimla and New Delhi weather; finance pools five IT and five pharma stock series without crossing stock boundaries; energy uses ETTh1 and ETTh2. Per-series chronological train, validation and test boundaries are retained. Validation and test inputs include historical context from the preceding split, while their forecast futures stay in their own split. Each domain's scaler is fitted only to its training segment. Per-domain target scaling itself uses observed target history, so Source-only means source weights with target-domain normalization.</p>
<p>The weather downloader requests hourly temperature, humidity, dew point, pressure, wind, cloud cover and shortwave radiation from the Open-Meteo archive for the two site coordinates, 2021 through 2024, in Asia/Kolkata time. Cached response elevations are 2195 m for Shimla and 214 m for Delhi. Neither cached raw hourly response contains null values. Finance downloads adjusted daily OHLCV, then derives log return as 100 ln(Close divided by previous Close), intraday log range as 100 ln(High divided by Low), overnight gap as 100 ln(Open divided by previous Close), and log volume as ln(1 plus Volume). Invalid rows and zero-volume days are excluded. The ten stock histories remain separate series. Energy downloads the hourly ETTh1 and ETTh2 CSV files directly from the dataset repository, retaining six load variables and oil temperature. Project-specific chronological fractions are 70/10/20 percent for climate and finance, and 60/20/20 percent for energy.</p>
<p>Raw and processed dataset paths and SHA256 checksums, package versions, seeds, budgets, candidate architectures and recipes are saved in each manifest. Climate provenance should be described as the cached Open-Meteo historical API response: the downloader does not pin an explicit reanalysis model, so the repository cannot establish an ERA5-only download merely from the intended label. TA verification of all three dataset choices is still required.</p>
<p>All new architecture, recipe and checkpoint decisions are frozen before test evaluation for that seed. No target future tensors are accepted by the new adaptation routine. Historical test results were already available when this extension was designed; the reused test period is therefore a development benchmark rather than a new prospective external holdout. Three seeds measure training variation, not generalization to new geographic locations or companies.</p>
<h2>Stage 3 Lightweight forecasting architectures</h2>
<p>The controlled sweep uses the original three-convolution LiteTCN, plus five- and seven-convolution versions with residual dilated blocks after the existing encoder. Extra blocks preserve temporal resolution. All LiteTCN versions retain RevIN, the temporal linear path and the original reconstruction head for checkpoint compatibility. Reconstruction is disabled in the new source objective, so the reported parameter count includes that unused head. This makes the depth comparison within the new sweep fair, while comparisons with old checkpoints also include a training-objective and training-budget change.</p>
<p>The patch MLP embeds non-overlapping temporal patches, mixes channels with a small MLP, projects the patch sequence to a latent state and combines a shared forecast head with a temporal linear path. It is a local alternative, not a reproduction of PatchTST or TSMixer. The DLinear-inspired baseline decomposes a normalized window with a moving average and applies shared linear seasonal and trend heads. RevIN is an explicit addition to the original DLinear design. Both alternatives are small enough for CPU training.</p>
<p>Every candidate uses the same source MSE objective, AdamW, learning rate 0.001, cosine learning-rate schedule, batch size 256, gradient clipping at 1 and 15 epochs. The lowest source validation MSE chooses the epoch and then the architecture independently for each seed. This selector optimizes multivariate source validation performance; it does not promise the lowest target temperature error.</p>
<h2>Stage 4 Adaptation loss and model selection</h2>
<p>For an observed target input of length L, the last H observed steps are withheld. The preceding L minus H steps are padded on the left by repeating their first value, and the model forecasts the withheld steps. These supervision values come entirely from the observed input history; the target future after the input window is never used. Padding creates an imperfect pretext distribution, which is a limitation of this lightweight approach.</p>
<pre>L = MSE(source forecast, source future)
    + 0.5 MSE(target inner forecast, withheld observed input)
    + 0.001 ||theta - theta_source||^2
    + optional 0.05 MK-MMD(source features, target features)</pre>
<p>The source replay term protects forecast behavior, the input-only forecast term adapts the dynamics, and the L2-SP term anchors the original weights. The optional MMD term tests whether discrepancy reduction adds value. DLinear aligns normalized forecast features; other models align their learned latent features. Both recipes start from the same validation-selected source checkpoint and use 8 adaptation rounds, AdamW at 0.0003 and 128-example source and target minibatches. Each round takes ceiling(number of target training windows divided by 256) updates, so the epoch label in adaptation logs is a fixed update budget rather than a full pass through all target windows.</p>
<pre>selection proxy = 0.5 source_validation_MSE / initial_source_validation_MSE
                + 0.5 target_input_pretext_MSE / initial_target_input_pretext_MSE</pre>
<p>Only source validation futures and a fixed probe of up to 512 target validation inputs contribute to this score. A checkpoint is eligible only if its source validation error rises by at most 10 percent. Epoch zero has score 1 and is always available. The lowest eligible score across both recipes and the source fallback selects the adapted model. This is a heuristic selection rule, not a guarantee against target negative transfer.</p>
<h2>Stage 5 Metrics and efficiency</h2>
<p>MSE and MAE use per-domain standardized units. Key RMSE and every per-variable RMSE use original measurement units. MASE_train averages each variable's forecast MAE divided by that variable's training seasonal absolute-difference scale. Seasonal naive, last-value persistence and the input-window mean provide no-training references. Parameter counts quantify storage and model size; source error after adaptation measures retention. Accuracy is not expressed as a percentage because this is continuous-valued regression. A supervised target fine-tune from the selected source is a label-using reference, not an upper bound that must always win.</p>
""" + efficiency + "\n".join(sections) + transfer_section + """
<h2>Stage 6 Interface methodology</h2>
<p>The Streamlit interface selects a dataset, experiment family and source or target domain. Architecture metadata reconstructs each saved model; saved training-only scalers transform uploads and restore predictions to original units. A user can upload one multivariate series, examine historical backtests with actual futures, or forecast after the last row when future observations are unavailable. Finance also accepts chronological raw OHLCV and derives the same features as training.</p>
<p>The interface rejects missing or nonfinite values, mixed series, duplicate dates and dates in reverse order. It does not fill a missing input using later observations. Inference is batched to bound memory usage. It plots observed inputs, actual future values and forecasts; target mode can overlay the unadapted source. It reports window error, rolling-file evaluation against seasonal naive and persistence, per-variable errors in original units and downloadable forecast CSVs. Short files with exactly L rows produce future forecasts without an invalid zero-range slider.</p>
<h2>Stage 7 Reproduction and submission</h2>
<pre>python -m src.improve --datasets climate energy finance --seeds 0 1 2 --epochs 15 --adapt-epochs 8
python -m src.transfer_selection --datasets climate energy finance --seeds 0 1 2
python -m src.check_improved
python -m src.benchmark_inference
python -m src.improvement_report
python -m unittest discover -s tests -v
streamlit run app/app.py</pre>
<p>The experiment command resumes completed candidate checkpoints and seeds when its manifest matches the saved settings. The deployable source and adapted weights plus architecture and scaler metadata are under models/improved/DATASET. Per-seed candidate checkpoints, adaptation checkpoints and supervised references are retained in seed directories. Results, validation decisions, adaptation histories, full test metrics and deterministic forecast examples are under results/improved. The first requested seed is the deployment seed; it is not chosen by test accuracy.</p>
<p>The original report and presentation describe the earlier experiment generation and remain historical artifacts. Use this stage update and its CSV files for the new findings. Before assessment, the team must obtain TA dataset approval, confirm team names and roll numbers, integrate this update into the final required report format, and rehearse both domain upload flows. The reported benchmarks do not establish reliable prediction of individual daily stock returns.</p>
<h2>References and dataset sources</h2>
<ul><li><a href="https://arxiv.org/abs/2205.13504">Zeng et al. Are Transformers Effective for Time Series Forecasting</a> — motivation for a decomposition linear baseline.</li>
<li><a href="https://arxiv.org/abs/2303.06053">Chen et al. TSMixer An All MLP Architecture for Time Series Forecasting</a> — motivation for testing lightweight MLP alternatives.</li>
<li><a href="https://open-meteo.com/en/docs/historical-weather-api">Open-Meteo Historical Weather API</a>.</li>
<li><a href="https://github.com/zhouhaoyi/ETDataset">ETT dataset repository</a>.</li>
<li><a href="https://github.com/ranaroussi/yfinance">yfinance data retrieval library</a>.</li></ul>
"""
    css = """body{max-width:1200px;margin:45px auto;padding:0 28px;font:16px/1.6 system-ui,sans-serif;color:#182330;background:#fff}h1{font-size:34px;line-height:1.2;max-width:920px}h2{margin-top:48px;font-size:25px}h3{margin-top:28px}.subtitle,figcaption{color:#536477}table{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow-x:auto;margin:22px 0}th,td{border:1px solid #d6dfe7;padding:9px 10px;text-align:right;white-space:normal}th{background:#edf2f6}th:first-child,td:first-child{text-align:left}tr:nth-child(even){background:#f8fafc}figure{margin:30px 0}img{max-width:100%;height:auto}figcaption{font-size:13px}pre{background:#f1f5f8;padding:18px;overflow-x:auto;font-size:13px}a{color:#1b617c}@media print{body{font-size:10pt;max-width:none;margin:0;padding:0}h2{break-after:avoid}figure{break-inside:avoid}table{font-size:8pt}pre{white-space:pre-wrap}}"""
    report = ROOT / "report" / "IE643_Improvement_Report.html"
    report.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IE643 forecasting experiment update</title><style>{css}</style></head><body>{body}</body></html>', encoding="utf-8")
    print(headline_df.to_string(index=False))
    print(f"\nReport: {report}")


if __name__ == "__main__":
    build()
