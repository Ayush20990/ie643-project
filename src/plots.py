"""Generate all report / slide figures from saved results.   python -m src.plots"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import DATASETS, RESULTS_DIR  # noqa: E402
from .data import load_domain  # noqa: E402

FIG = RESULTS_DIR / "figures"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"
METHOD_COLORS = {"care": BLUE, "coral": AQUA, "mmd": VIOLET, "dann": YELLOW}
NAMES = {"seasonal_naive": "Seasonal naive", "source_only": "Source-only", "dann": "DANN", "coral": "Deep CORAL",
         "mmd": "MMD", "care_noalign": "CARE − align", "care_nossl": "CARE − SSL", "care_noanchor": "CARE − anchor",
         "care": "CARE-DA v1", "care_v2": "CARE-DA v2 (ours)", "target_only": "Target-only*",
         "oracle_finetune": "Fine-tune oracle*"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 150,
    "lines.linewidth": 2, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def adaptation_curves(ds, hist, summ):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    oracle = summ.loc["oracle_finetune", "MSE_mean"]
    for m, c in METHOD_COLORS.items():
        h = pd.DataFrame(hist[m])
        lw = 2.6 if m == "care" else 1.6
        axs[0].plot(h.epoch, h.tgt_test_mse, color=c, lw=lw, label=NAMES[m], marker="o", ms=3)
        axs[1].plot(h.epoch, h.mmd_feat, color=c, lw=lw, label=NAMES[m], marker="o", ms=3)
        axs[2].plot(h.epoch, h.src_val_mse, color=c, lw=lw, label=NAMES[m], marker="o", ms=3)
    axs[0].axhline(oracle, color=INK2, ls=":", lw=1.2)
    axs[0].text(0.2, oracle, " fine-tune oracle (uses labels)", va="bottom", fontsize=8, color=INK2)
    axs[0].set_title("Target test MSE ↓ (monitor only)")
    axs[1].set_title("Feature discrepancy MMD² ↓")
    axs[2].set_title("Source validation MSE (forgetting)")
    for a in axs:
        a.set_xlabel("adaptation epoch")
    axs[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(DATASETS[ds]["title"], x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / f"{ds}_adaptation_curves.png")
    plt.close(fig)


def loss_components(ds, hist):
    h = pd.DataFrame(hist["care"]).iloc[1:]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    comps = [("sup", "source forecast MSE", INK), ("ssl", "target masked-recon (SSL)", BLUE),
             ("mmd", "MK-MMD", VIOLET), ("anchor", "L2-SP anchor", ORANGE), ("coral", "CORAL ×100", AQUA)]
    for k, lab, c in comps:
        if k in h:
            ax.plot(h.epoch, h[k] * (100 if k == "coral" else 1), label=lab, color=c, marker="o", ms=3)
    ax.set_xlabel("adaptation epoch")
    ax.set_ylabel("weighted loss term")
    ax.set_title(f"CARE-DA loss components — {ds}")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(FIG / f"{ds}_loss_components.png")
    plt.close(fig)


def latent_pca(ds):
    lat = np.load(RESULTS_DIR / ds / "latents.npz")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, (m, title) in zip(axs, [("source_only", "Before adaptation (source model)"),
                                     ("care", "After CARE-DA")]):
        zs, zt = lat[f"{m}_zs"], lat[f"{m}_zt"]
        Z = np.concatenate([zs, zt])
        Z = Z - Z.mean(0)
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        P = Z @ vt[:2].T
        n = len(zs)
        ax.scatter(P[:n, 0], P[:n, 1], s=9, color=BLUE, alpha=0.55, label="Domain 1 (source)", linewidths=0)
        ax.scatter(P[n:, 0], P[n:, 1], s=9, color=ORANGE, alpha=0.55, label="Domain 2 (target)", linewidths=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].legend(loc="best", fontsize=8, markerscale=2)
    fig.suptitle(f"Latent space (PCA of encoder features) — {DATASETS[ds]['title']}", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"{ds}_latent_pca.png")
    plt.close(fig)


def forecast_example(ds):
    cfg = DATASETS[ds]
    f = np.load(RESULTS_DIR / ds / "forecasts.npz")
    k = cfg["columns"].index(cfg["key"])
    Xt, Yt = f["Xt"], f["Yt"]
    err = ((f["Pt_source_only"] - Yt) ** 2).mean((1, 2)) - ((f["Pt_care"] - Yt) ** 2).mean((1, 2))
    i = int(np.argsort(err)[int(0.75 * len(err))])  # a representative (not the best) window
    L, H = cfg["L"], cfg["H"]
    show_L = min(L, 72)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    t_in, t_out = np.arange(-show_L, 0), np.arange(0, H)
    ax.plot(t_in, Xt[i, -show_L:, k], color=MUTED, label="input (look-back)")
    ax.plot(t_out, Yt[i, :, k], color=INK, label="actual")
    ax.plot(t_out, f["Pt_source_only"][i, :, k], color=ORANGE, ls="--", label="source-only")
    ax.plot(t_out, f["Pt_care"][i, :, k], color=BLUE, label="CARE-DA (ours)")
    ax.plot(t_out, f["Pt_oracle_finetune"][i, :, k], color=AQUA, ls=":", label="fine-tune oracle")
    ax.axvline(0, color=GRID, lw=1.5)
    ax.set_xlabel("time step relative to forecast origin")
    ax.set_ylabel(f"{cfg['key']} (z-scored)")
    ax.set_title(f"Domain 2 forecast example — {cfg['target_label']}")
    ax.legend(fontsize=8, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.2))
    fig.savefig(FIG / f"{ds}_forecast_example.png")
    plt.close(fig)


def method_bars(ds, summ):
    s = summ.dropna(subset=["MSE_mean"])
    names = [NAMES.get(m, m) for m in s.index]
    colors = [BLUE if m == "care" else (MUTED if m in ("target_only", "oracle_finetune", "seasonal_naive")
                                        else "#b9b8b1") for m in s.index]
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    y = np.arange(len(s))[::-1]
    ax.barh(y, s.MSE_mean, xerr=s.MSE_std.fillna(0), color=colors, height=0.62,
            error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    ax.set_yticks(y, names)
    lo = s.MSE_mean.min() * 0.9
    ax.set_xlim(lo, s.MSE_mean.max() * 1.04)
    for yy, v in zip(y, s.MSE_mean):
        ax.text(v, yy, f" {v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("target-domain test MSE (z-scored) ↓   (* uses target labels)")
    ax.grid(axis="y", visible=False)
    ax.set_title(DATASETS[ds]["title"])
    fig.savefig(FIG / f"{ds}_method_bars.png")
    plt.close(fig)


def domain_shift(ds):
    cfg = DATASETS[ds]
    k = cfg["key"]
    s, t = load_domain(ds, cfg["source"]), load_domain(ds, cfg["target"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw=dict(width_ratios=[2.2, 1]))
    if "series" in s:
        s, t = s[s.series == s.series.iloc[0]], t[t.series == t.series.iloc[0]]
    n = {"climate": 24 * 14, "energy": 24 * 14, "finance": 250}[ds]
    for d, c, lab in [(s, BLUE, cfg["source_label"]), (t, ORANGE, cfg["target_label"])]:
        seg = d.iloc[len(d) // 2: len(d) // 2 + n]
        axs[0].plot(np.arange(n), seg[k].values, color=c, lw=1.4, label=lab)
        axs[1].hist(d[k].values, bins=60, color=c, alpha=0.55, density=True, label=lab)
    axs[0].set_title(f"{k}: same calendar window, two domains")
    axs[0].set_xlabel("time step (" + ("trading days" if ds == "finance" else "hours") + ")")
    axs[0].set_ylabel(f"{k} [{cfg['key_unit']}]")
    axs[1].set_title(f"Marginal distribution of {k}")
    axs[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"{ds}_domain_shift.png")
    plt.close(fig)


def cross_dataset():
    rows = []
    methods = ["dann", "coral", "mmd", "care_noalign", "care_nossl", "care_noanchor", "care", "oracle_finetune"]
    for ds in DATASETS:
        p = RESULTS_DIR / ds / "summary_target.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p, index_col=0)
        so, orc = s.loc["source_only", "MSE_mean"], s.loc["oracle_finetune", "MSE_mean"]
        for m in methods:
            rows.append({"dataset": ds, "method": m, "rel_change_%": 100 * (s.loc[m, "MSE_mean"] / so - 1),
                         "GCR_of_means": (so - s.loc[m, "MSE_mean"]) / (so - orc)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "cross_dataset_summary.csv", index=False)
    dss = df.dataset.unique()
    fig, axs = plt.subplots(1, len(dss), figsize=(4.3 * len(dss), 3.8))
    axs = np.atleast_1d(axs)
    for ax, ds in zip(axs, dss):
        g = df[df.dataset == ds].set_index("method").reindex(methods)
        y = np.arange(len(g))[::-1]
        cols = [MUTED if m == "oracle_finetune" else (BLUE if m.startswith("care") else "#b9b8b1") for m in g.index]
        ax.barh(y, g["rel_change_%"], color=cols, height=0.62)
        for yy, v in zip(y, g["rel_change_%"]):
            ax.text(v, yy, f" {v:+.1f}%" if v >= 0 else f"{v:+.1f}% ", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=INK2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y, [NAMES[m] for m in g.index])
        lim = max(abs(g["rel_change_%"]).max() * 1.75, 3)
        ax.set_xlim(-lim, lim)
        ax.set_title(ds.capitalize())
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("target MSE vs source-only (%)  ← better")
    fig.suptitle("Change in Domain 2 test MSE relative to the un-adapted source model (mean of 3 seeds; * uses labels)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "gap_closure_all.png")
    plt.close(fig)


def sensitivity():
    res = {ds: json.loads((RESULTS_DIR / ds / "sensitivity.json").read_text())
           for ds in DATASETS if (RESULTS_DIR / ds / "sensitivity.json").exists()}
    if not res:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    colors = {"climate": BLUE, "energy": ORANGE, "finance": AQUA}
    for ax, (key, lab) in zip(axs, [("lambda_R", "SSL weight λ_R"), ("lambda_A", "anchor weight λ_A")]):
        for ds, r in res.items():
            base = r["source_only"]["tgt"]
            xs = np.arange(len(r[key]))
            ax.plot(xs, [100 * (row["tgt_test_mse"] / base - 1) for row in r[key]], marker="o", ms=5,
                    color=colors[ds], label=ds)
            ax.set_xticks(xs, [f"{row['value']:g}" for row in r[key]])
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xlabel(lab + "  (others at default)")
        ax.set_ylabel("target MSE vs source-only (%)  ↓")
    axs[0].legend(fontsize=8)
    fig.suptitle("Post-hoc sensitivity of CARE-DA (seed 0; not used for model selection)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity.png")
    plt.close(fig)


def care_v2_bars():
    p = RESULTS_DIR / "care_v2_compare.csv"
    if not p.exists():
        return
    df = pd.read_csv(p, index_col=0)
    dss = [d for d in DATASETS if d in set(df.dataset)]
    if not dss:
        return
    fig, axs = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3.4))
    axs = np.atleast_1d(axs)
    colors = {"source_only": MUTED, "care_v1": "#b9b8b1", "care_v2": BLUE}
    labels = {"source_only": "Source-only", "care_v1": "CARE-DA v1", "care_v2": "CARE-DA v2"}
    for ax, ds in zip(axs, dss):
        g = df[df.dataset == ds]
        y = np.arange(len(g))[::-1]
        ax.barh(y, g["rel_change_%"], color=[colors.get(i, MUTED) for i in g.index], height=0.62)
        for yy, v in zip(y, g["rel_change_%"]):
            ax.text(v, yy, f" {v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=INK2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y, [labels.get(m, m) for m in g.index])
        ax.set_title(ds.capitalize())
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("target MSE vs source-only (%)  ← better")
    fig.suptitle("CARE-DA v2 vs v1 (seed 0; no target labels)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "care_v2_compare.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        domain_shift(ds)
        p = RESULTS_DIR / ds / "summary_target.csv"
        if not p.exists():
            continue
        summ = pd.read_csv(p, index_col=0)
        hist = json.loads((RESULTS_DIR / ds / "histories.json").read_text())
        adaptation_curves(ds, hist, summ)
        loss_components(ds, hist)
        latent_pca(ds)
        forecast_example(ds)
        method_bars(ds, summ)
    cross_dataset()
    sensitivity()
    care_v2_bars()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
