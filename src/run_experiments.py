"""Run the full benchmark for one dataset.

Usage:  python -m src.run_experiments --dataset climate --seeds 0 1 2

For each seed:
  1. train LiteTCN on Domain 1 (source) with forecasting + masked-reconstruction loss
  2. evaluate zero-shot on Domain 2 (source-only)
  3. run every unsupervised adaptation recipe in train.METHODS from that same checkpoint
  4. train two label-using references on Domain 2: fine-tune oracle and target-only
Saves metrics, adaptation logs, latents, forecasts and the seed-0 checkpoints (source + CARE-DA adapted).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from .config import MODEL_DIR, RESULTS_DIR, SAMPLE_DIR
from .data import load_domain, prepare, save_meta
from .losses import mmd_rbf
from .metrics import all_metrics, gap_closure, seasonal_naive
from .model import n_params
from .train import (METHODS, adapt, build_model, embed, predict, set_seed, train_supervised)

SRC_EPOCHS, ADAPT_EPOCHS, FT_EPOCHS = 20, 10, 15
ADAPT_LR = 3e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    ds = args.dataset
    torch.set_num_threads(4)

    out = RESULTS_DIR / ds
    out.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / ds).mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_log.txt", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    data = prepare(ds)
    cfg = data["cfg"]
    save_meta(ds, data, MODEL_DIR / ds / "meta.json")
    key_idx = cfg["columns"].index(cfg["key"])
    S, T = data["source"], data["target"]
    log(f"== {cfg['title']}  L={cfg['L']} H={cfg['H']} C={len(cfg['columns'])}")
    log(f"windows source train/val/test {len(S['train'][0])}/{len(S['val'][0])}/{len(S['test'][0])}; "
        f"target {len(T['train'][0])}/{len(T['val'][0])}/{len(T['test'][0])}")

    def metrics(model_or_pred, dom):
        X, Y = dom["test"]
        P = model_or_pred if isinstance(model_or_pred, np.ndarray) else predict(model_or_pred, X)
        return all_metrics(X.numpy(), Y.numpy(), P, cfg["season"], dom["scaler"], key_idx)

    # seasonal-naive reference (no learning)
    naive = {r: metrics(seasonal_naive(d["test"][0].numpy(), cfg["H"], cfg["season"]), d)
             for r, d in [("source", S), ("target", T)]}
    records = []
    for r, m in naive.items():
        records.append({"seed": -1, "method": "seasonal_naive", "eval_domain": r, **m})

    probe_s, probe_t = S["test"][0][:1500], T["test"][0][:1500]

    for seed in args.seeds:
        t0 = time.time()
        set_seed(seed)
        src_model = build_model(cfg)
        if seed == args.seeds[0]:
            log(f"LiteTCN params: {n_params(src_model):,}")
        src_model, hist_src = train_supervised(src_model, S["train"], S["val"], SRC_EPOCHS, 1e-3,
                                               seed=seed, log=log, tag=f"source s{seed}")
        models = {"source_only": src_model}
        hists = {"source_train": hist_src}
        for name, recipe in METHODS.items():
            models[name], hists[name] = adapt(src_model, S, T, recipe, ADAPT_EPOCHS, ADAPT_LR, seed=seed,
                                              log=log, tag=f"{name} s{seed}", monitor=T["test"])
        set_seed(seed)
        oracle = build_model(cfg)
        oracle.load_state_dict(src_model.state_dict())
        models["oracle_finetune"], hists["oracle_finetune"] = train_supervised(
            oracle, T["train"], T["val"], FT_EPOCHS, 5e-4, seed=seed, log=log, tag=f"oracle s{seed}")
        set_seed(seed)
        models["target_only"], hists["target_only"] = train_supervised(
            build_model(cfg), T["train"], T["val"], SRC_EPOCHS, 1e-3, seed=seed, log=log, tag=f"tgt-only s{seed}")

        for name, m in models.items():
            zs, zt = embed(m, probe_s), embed(m, probe_t)
            feat_mmd = float(mmd_rbf(zs, zt))
            for r, d in [("source", S), ("target", T)]:
                records.append({"seed": seed, "method": name, "eval_domain": r, **metrics(m, d),
                                "feat_mmd": feat_mmd})
        log(f"seed {seed} done in {time.time() - t0:.0f}s")

        if seed == args.seeds[0]:
            torch.save(models["source_only"].state_dict(), MODEL_DIR / ds / "source_model.pt")
            torch.save(models["care"].state_dict(), MODEL_DIR / ds / "adapted_model.pt")
            (out / "histories.json").write_text(json.dumps(hists, indent=1))
            lat = {}
            for name in ["source_only", "care", "coral", "dann"]:
                lat[f"{name}_zs"] = embed(models[name], probe_s[:600]).numpy()
                lat[f"{name}_zt"] = embed(models[name], probe_t[:600]).numpy()
            np.savez_compressed(out / "latents.npz", **lat)
            Xt, Yt = T["test"]
            Xs, Ys = S["test"]
            np.savez_compressed(out / "forecasts.npz", Xt=Xt.numpy(), Yt=Yt.numpy(), Xs=Xs.numpy(), Ys=Ys.numpy(),
                                **{f"Pt_{n}": predict(models[n], Xt) for n in
                                   ["source_only", "care", "oracle_finetune", "coral", "dann"]},
                                Ps_source_only=predict(models["source_only"], Xs),
                                Ps_care=predict(models["care"], Xs))

    df = pd.DataFrame(records)
    df.to_csv(out / "metrics_all.csv", index=False)
    summarize(df, out, log)
    make_samples(ds, cfg)
    logf.close()


def summarize(df, out, log):
    tgt = df[df.eval_domain == "target"]
    rows = []
    for seed in sorted(tgt.seed.unique()):
        if seed < 0:
            continue
        g = tgt[tgt.seed == seed].set_index("method")
        src = df[(df.seed == seed) & (df.eval_domain == "source")].set_index("method")
        for m in g.index:
            rows.append({"seed": seed, "method": m,
                         "GCR_MSE": gap_closure(g.loc["source_only", "MSE"], g.loc[m, "MSE"],
                                                g.loc["oracle_finetune", "MSE"]),
                         "source_MSE_after": src.loc[m, "MSE"],
                         "forgetting_%": 100 * (src.loc[m, "MSE"] / src.loc["source_only", "MSE"] - 1)})
    extra = pd.DataFrame(rows)
    merged = tgt.merge(extra, on=["seed", "method"], how="left")
    cols = ["MSE", "MAE", "MASE", "key_RMSE", "key_MAE", "feat_mmd", "GCR_MSE", "forgetting_%"]
    summ = merged.groupby("method")[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    order = ["seasonal_naive", "source_only", "dann", "coral", "mmd", "care_noalign", "care_nossl",
             "care_noanchor", "care", "care_v2", "target_only", "oracle_finetune"]
    summ = summ.reindex([o for o in order if o in summ.index])
    summ.to_csv(out / "summary_target.csv")
    log("\nTarget-domain test results (mean over seeds):")
    log(summ[[c for c in summ.columns if c.endswith("_mean")]].round(4).to_string())


def make_samples(ds, cfg):
    """Raw (unscaled) CSVs from the held-out test period of each domain for the interface."""
    SAMPLE_DIR.mkdir(exist_ok=True)
    n = {"climate": 24 * 21, "energy": 24 * 21, "finance": 250}[ds]
    for role, dom in [("domain1", cfg["source"]), ("domain2", cfg["target"])]:
        if ds == "finance":
            from .config import FINANCE_TICKERS, RAW_DIR
            t = FINANCE_TICKERS[dom][1 if role == "domain1" else 0]
            raw = pd.read_csv(RAW_DIR / "finance" / f"{t}.csv")
            raw.tail(n).to_csv(SAMPLE_DIR / f"{ds}_{role}_{t.replace('.NS', '')}_ohlcv.csv", index=False)
        else:
            df = load_domain(ds, dom)
            df[["date"] + cfg["columns"]].tail(n).to_csv(SAMPLE_DIR / f"{ds}_{role}_{dom}.csv", index=False)


if __name__ == "__main__":
    main()
