"""Train CARE-DA v2 from the saved Domain-1 checkpoints (no source retrain).

v2 drops masked reconstruction (ablation/sensitivity: it hurts on every benchmark)
and adds forecast-consistent pretext, multi-scale MMD and spectral alignment.

Usage:  python -m src.adapt_v2
        python -m src.adapt_v2 --dataset climate
"""
import argparse
import json
import shutil
import time

import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, RESULTS_DIR
from .data import prepare
from .metrics import all_metrics
from .run_experiments import ADAPT_EPOCHS, ADAPT_LR
from .train import METHODS, adapt, build_model, eval_mse, predict


def run_one(ds, log):
    data = prepare(ds)
    cfg = data["cfg"]
    S, T = data["source"], data["target"]
    key_idx = cfg["columns"].index(cfg["key"])
    src = build_model(cfg)
    src.load_state_dict(torch.load(MODEL_DIR / ds / "source_model.pt", map_location="cpu", weights_only=True))
    src.eval()

    log(f"== {cfg['title']}")
    t0 = time.time()
    model, hist = adapt(
        src, S, T, METHODS["care_v2"], ADAPT_EPOCHS, ADAPT_LR, seed=0,
        log=log, tag=f"care_v2 {ds}", monitor=T["test"], select="src_mmd",
    )
    log(f"care_v2 {ds} done in {time.time() - t0:.0f}s")

    dest = MODEL_DIR / ds
    dest.mkdir(parents=True, exist_ok=True)
    v1 = dest / "adapted_model.pt"
    bak = dest / "adapted_model_v1.pt"
    if v1.exists() and not bak.exists():
        shutil.copy2(v1, bak)
    torch.save(model.state_dict(), dest / "adapted_model_v2.pt")
    torch.save(model.state_dict(), v1)

    def pack(m, dom):
        X, Y = dom["test"]
        return all_metrics(X.numpy(), Y.numpy(), predict(m, X), cfg["season"], dom["scaler"], key_idx)

    v1_model = build_model(cfg)
    if bak.exists():
        v1_model.load_state_dict(torch.load(bak, map_location="cpu", weights_only=True))
    else:
        v1_model.load_state_dict(src.state_dict())

    rows = {
        "source_only": pack(src, T),
        "care_v1": pack(v1_model, T),
        "care_v2": pack(model, T),
    }
    rows["source_only"]["src_MSE"] = eval_mse(src, *S["test"])
    rows["care_v1"]["src_MSE"] = eval_mse(v1_model, *S["test"])
    rows["care_v2"]["src_MSE"] = eval_mse(model, *S["test"])
    so = rows["source_only"]["MSE"]
    for name, r in rows.items():
        r["rel_change_%"] = 100 * (r["MSE"] / so - 1)
    table = pd.DataFrame(rows).T
    log("\n" + table.round(4).to_string())
    out = RESULTS_DIR / ds
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "care_v2_compare.csv")
    (out / "care_v2_history.json").write_text(json.dumps(hist, indent=1))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    torch.set_num_threads(4)

    def log(msg):
        print(msg, flush=True)

    for ds in args.dataset:
        run_one(ds, log)
    frames = []
    for ds in DATASETS:
        p = RESULTS_DIR / ds / "care_v2_compare.csv"
        if p.exists():
            frames.append(pd.read_csv(p, index_col=0).assign(dataset=ds))
    if frames:
        pd.concat(frames).to_csv(RESULTS_DIR / "care_v2_compare.csv")
    log("\nWrote models/*/adapted_model_v2.pt and results/care_v2_compare.csv")


if __name__ == "__main__":
    main()
