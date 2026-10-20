"""Measure warm CPU forecast latency and deployment checkpoint size.

Run after training: python -m src.benchmark_inference
Timing excludes CSV loading, scaling, checkpoint loading and interface rendering.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, RESULTS_DIR
from .inference import load_bundle, load_dlinear_bundle
from .model import n_params


@torch.inference_mode()
def timing(model, x, repeats):
    for _ in range(10):
        model(x)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        elapsed.append(1000 * (time.perf_counter() - start))
    return {"median_ms": float(np.median(elapsed)), "p95_ms": float(np.quantile(elapsed, .95))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()
    if args.threads < 1 or args.repeats < 10:
        ap.error("Use at least one CPU thread and 10 repeats")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rows = []
    for ds, cfg in DATASETS.items():
        if not (MODEL_DIR / "improved" / ds / "meta.json").exists():
            continue
        variants = [(False, False, "original_care_v2"), (True, False, "selected_adapted")]
        if (MODEL_DIR / "transfer_selection" / ds / "meta.json").exists():
            variants.append((True, True, "history_selected_adapted"))
            variants.append((True, "dlinear", "dlinear_adapted"))
        for improved, transfer, label in variants:
            if transfer == "dlinear":
                meta, models = load_dlinear_bundle(ds)
                checkpoint = Path(meta["_checkpoint_path"])
            else:
                _, models = load_bundle(ds, improved, transfer)
                path = (MODEL_DIR / "transfer_selection" / ds if transfer else
                        MODEL_DIR / "improved" / ds if improved else MODEL_DIR / ds)
                checkpoint = path / "adapted_model.pt"
            model = models["adapted_model"]
            for bs in (1, 64):
                # Same tensor shapes and seeded synthetic inputs; this benchmark measures runtime only.
                x = torch.randn(bs, cfg["L"], len(cfg["columns"]))
                row = {"dataset": ds, "model": label, "batch_size": bs,
                       "params": n_params(model), "checkpoint_KiB": checkpoint.stat().st_size / 1024,
                       **timing(model, x, args.repeats)}
                row["median_ms_per_window"] = row["median_ms"] / bs
                rows.append(row)
    if not rows:
        raise RuntimeError("No deployment bundles found")
    out = RESULTS_DIR / "improved"
    pd.DataFrame(rows).to_csv(out / "inference_latency.csv", index=False)
    metadata = {"threads": args.threads, "repeats": args.repeats, "warmups": 10,
                "torch": str(torch.__version__), "processor": platform.processor(),
                "platform": platform.platform(), "timing": "warm forward only, no IO or scaling"}
    (out / "inference_latency_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
