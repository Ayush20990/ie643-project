"""Source-only load/temperature diagnostics, not evidence of causal effects.

No target-domain data are read. All explanatory features are known at the forecast
origin; the actual future loads never become predictors. Source validation is
used for ridge selection and is explicitly not an independent evaluation.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, PROC_DIR, RESULTS_DIR
from .data import Scaler, chrono_split, load_domain, make_windows
from .improve import write_json

PENALTIES = (.0001, .001, .01, .1, 1., 10., 100.)


def features(x, clock, with_loads):
    temp = x[..., -1]
    last = temp[:, -1:]
    parts = [temp-last, last, clock]
    if with_loads:
        load = x[..., :-1]
        parts += [load[:, -lag-1] for lag in (0, 1, 3, 6, 12, 23)]
        parts += [load[:, -width:].mean(1) for width in (3, 6, 12, 24)]
        parts += [load[:, -1]-load[:, -width] for width in (3, 6, 12, 24)]
    return np.concatenate(parts, -1).astype(np.float64)


def ridge_forecasts(train_x, train_y, val_x):
    mean, sd = train_x.mean(0), train_x.std(0).clip(.01)
    a, b = (train_x-mean)/sd, (val_x-mean)/sd
    target_mean = train_y.mean(0)
    gram, rhs = a.T@a/len(a), a.T@(train_y-target_mean)/len(a)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected = eigenvectors.T@rhs
    for penalty in PENALTIES:
        weight = eigenvectors@(projected/(eigenvalues.clip(0)[:, None]+penalty))
        yield penalty, b@weight+target_mean


def main():
    torch.set_num_threads(2)
    cfg = DATASETS["energy"]
    df = load_domain("energy", cfg["source"]).sort_values("date")
    values = df[cfg["columns"]].to_numpy(dtype=np.float32)
    parts = chrono_split([values], cfg["split"], cfg["L"])
    scaler = Scaler.fit(parts["train"])
    # Deliberately construct only source train/val windows; target/test are absent.
    windows = {s: tuple(t.numpy() for t in make_windows([scaler.transform(parts[s][0])], cfg["L"], cfg["H"]))
               for s in ("train", "val")}
    a, b = int(len(df)*cfg["split"][0]), int(len(df)*sum(cfg["split"][:2]))
    clocks = {}
    for split, dates in (("train", df.date.iloc[:a]), ("val", df.date.iloc[a-cfg["L"]:b])):
        origin = pd.DatetimeIndex(dates.iloc[cfg["L"]-1:cfg["L"]-1+len(windows[split][0])]).as_unit("ns")
        hours = origin.asi8.astype(np.float64)/3.6e12
        clocks[split] = np.stack([f(2*np.pi*hours/period) for period in (24,168) for f in (np.sin,np.cos)],-1)
    xt, yt = windows["train"]
    xv, yv = windows["val"]
    residual = yt[..., -1]-xt[:, -1:, -1]
    out = RESULTS_DIR/"progressive_litetcn"/"energy_drivers"
    out.mkdir(parents=True, exist_ok=True)
    searches, rows = [], []
    for name, loads in (("temperature_history_calendar", False), ("temperature_history_calendar_past_loads", True)):
        best = None
        for penalty, forecast_delta in ridge_forecasts(features(xt, clocks["train"], loads), residual,
                                                       features(xv, clocks["val"], loads)):
            prediction = forecast_delta+xv[:, -1:, -1]
            error = (prediction-yv[..., -1])*scaler.std[-1]
            row = {"model": name, "ridge": penalty, "source_val_24h_RMSE": float(np.sqrt(np.mean(error**2)))}
            searches.append(row)
            if best is None or row["source_val_24h_RMSE"] < best[0]["source_val_24h_RMSE"]:
                best = row, error
        row, error = best
        rows.append({**row, **{f"RMSE_first_{h}h": float(np.sqrt(np.mean(error[:, :h]**2))) for h in (1,3,6,12,24)}})
    # Association diagnostics use source training rows only, never future-load features.
    train = values[:a]
    delta = np.diff(train[:, -1])
    correlations = []
    for c, name in enumerate(cfg["columns"][:-1]):
        for lag in (0,1,3,6,12,24,48):
            # load at t-lag versus OT(t+1)-OT(t), both within source train.
            load, change = train[:len(delta)-lag, c], delta[lag:]
            corr = float(np.corrcoef(load, change)[0,1])
            correlations.append({"load": name, "past_lag_hours": lag, "correlation_with_next_OT_change": corr})
    pd.DataFrame(rows).to_csv(out/"source_validation_forecasts.csv", index=False)
    pd.DataFrame(searches).to_csv(out/"ridge_search.csv", index=False)
    pd.DataFrame(correlations).to_csv(out/"training_lag_associations.csv", index=False)
    write_json(out/"protocol.json", {"domain": "ETTh1 source only", "target_values_used": False,
        "future_load_inputs_used": False, "forecast": "direct 24-hour oil-temperature change from last observation",
        "source_train_rows": a, "source_val_rows": b-a, "penalties": PENALTIES,
        "selection_and_reporting_period": "source validation; not independent confirmation or causal proof",
        "source_file_sha256": hashlib.sha256((PROC_DIR/"energy"/"etth1.csv").read_bytes()).hexdigest()})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
