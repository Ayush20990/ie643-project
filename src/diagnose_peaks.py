"""Evaluation-only peak/timing diagnosis of frozen context ensembles.

Target forecasts here are descriptive development-test diagnostics. They do not
feed training or model selection. Source fitting diagnostics use source labels.
"""
import json

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, RESULTS_DIR
from .data import prepare
from .inference import load_strict_bundle
from .train import predict


def peak_metrics(truth, prediction):
    y, p = np.asarray(truth), np.asarray(prediction)
    index = np.arange(len(y))
    high, low = y.argmax(1), y.argmin(1)
    error = np.abs(p - y)
    interior = np.ones_like(error, dtype=bool)
    interior[index, high] = False
    interior[index, low] = False
    return {"RMSE": float(np.sqrt(np.mean((p-y)**2))), "MAE": float(error.mean()),
            "MAE_at_actual_peak": float(error[index, high].mean()),
            "MAE_at_actual_trough": float(error[index, low].mean()),
            "MAE_other_steps": float(error[interior].mean()),
            "max_value_bias": float((p.max(1)-y.max(1)).mean()),
            "min_value_bias": float((p.min(1)-y.min(1)).mean()),
            "mean_level_bias": float((p.mean(1)-y.mean(1)).mean()),
            "mean_actual_range": float(np.ptp(y, axis=1).mean()),
            "mean_predicted_range": float(np.ptp(p, axis=1).mean()),
            "mean_absolute_peak_timing_steps": float(np.abs(p.argmax(1)-high).mean()),
            "ramp_MAE": float(np.abs(np.diff(p, axis=1)-np.diff(y, axis=1)).mean())}


def main():
    torch.set_num_threads(4)
    out = RESULTS_DIR / "peak_diagnosis"
    out.mkdir(exist_ok=True)
    rows, examples = [], []
    for ds in ("climate", "energy"):
        data = prepare(ds)
        cfg = data["cfg"]
        _, models = load_strict_bundle(ds, context=True)
        j = cfg["columns"].index(cfg["key"])
        for role, split in (("source", "train"), ("source", "val"), ("source", "test"), ("target", "test")):
            x, y = data[role][split]
            if split == "train":
                indices = np.linspace(0, len(x)-1, min(4096, len(x)), dtype=int)
                x, y = x[indices], y[indices]
            p = predict(models["adapted_model" if role == "target" else "source_model"], x, bs=256)
            sd, mu = data[role]["scaler"].std[j], data[role]["scaler"].mean[j]
            metrics = peak_metrics(y.numpy()[..., j] * sd + mu, p[..., j] * sd + mu)
            rows.append({"dataset": ds, "domain": role, "split": split, "windows": len(x), **metrics})
        a = np.load(RESULTS_DIR / "strict_context" / ds / "forecast_examples.npz")
        for i in (0, 128, 255):
            y = a["target_Y"][i, :, j]*sd+mu
            p = a["target_strict_adapted"][i, :, j]*sd+mu
            examples.append({"dataset": ds, "test_window": int(a["target_indices"][i]),
                             "actual_peak": float(y.max()), "predicted_peak": float(p.max()),
                             "actual_trough": float(y.min()), "predicted_trough": float(p.min()),
                             "actual_peak_step": int(y.argmax()), "predicted_peak_step": int(p.argmax())})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "diagnostics.csv", index=False)
    pd.DataFrame(examples).to_csv(out / "plotted_examples.csv", index=False)
    print(frame.to_string(index=False))
    print(pd.DataFrame(examples).to_string(index=False))


if __name__ == "__main__":
    main()
