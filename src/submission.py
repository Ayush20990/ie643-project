"""Strict, frozen-model workflows shared by the assessment app and its tests."""
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASETS, RESULTS_DIR
from .data import Scaler
from .inference import (forecast_batches, load_bundle, load_event_boost_bundle,
                        load_next_hour_bundle, load_next_hour_reference_bundle,
                        load_peak_bundle, load_peak_next_hour_bundle,
                        load_residual_bundle, load_strict_bundle, read_forecast_csv)
from .metrics import all_metrics, seasonal_naive


def options(ds, horizon):
    if ds not in DATASETS or horizon not in ([5] if ds == "finance" else [1,24]):
        raise ValueError("Unsupported dataset/forecast horizon")
    if horizon == 1:
        return ["Peak-calibrated hybrid", "LiteTCN + local dynamics", "Small next-hour LiteTCN", "Earlier hybrid reference"]
    return ["Strict forecasting hybrid", "LiteTCN + geometry boosting", "Retained LiteTCN", "Original LiteTCN / CARE-DA v2"] if ds != "finance" else ["Strict forecasting hybrid", "Original LiteTCN / CARE-DA v2"]


def load_selected(ds, horizon, name):
    if name not in options(ds, horizon):
        raise ValueError("Model is unavailable for this dataset/horizon")
    if horizon == 1:
        if name == "Peak-calibrated hybrid":
            return load_peak_next_hour_bundle(ds)
        if name == "Earlier hybrid reference":
            return load_next_hour_reference_bundle(ds)
        return load_next_hour_bundle(ds, hybrid=name == "LiteTCN + local dynamics")
    if name == "Strict forecasting hybrid":
        return load_strict_bundle(ds, context=ds == "climate")
    if name == "LiteTCN + geometry boosting":
        return load_event_boost_bundle(ds)
    if name == "Retained LiteTCN":
        return load_peak_bundle(ds) if ds == "climate" else load_residual_bundle(ds)
    return load_bundle(ds)


def read_input(file, ds):
    """Check raw headers before pandas can silently rename duplicate columns."""
    raw = Path(file).read_bytes() if isinstance(file, (str, Path)) else file.getvalue() if hasattr(file, "getvalue") else file.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw:
        raise ValueError("The uploaded file is empty.")
    try:
        header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    except (UnicodeError, StopIteration, csv.Error) as exc:
        raise ValueError("Upload a UTF-8 comma-separated CSV file.") from exc
    clean = [c.strip() for c in header]
    if len(set(clean)) != len(clean):
        raise ValueError("CSV contains duplicate column names.")
    frame = read_forecast_csv(io.BytesIO(raw), ds, DATASETS[ds]["columns"])
    if ds != "finance":
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Hourly data needs a date or timestamp column.")
        if not frame.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError("Climate and energy data must contain consecutive hourly rows.")
    return frame


def scale_frame(frame, meta, role):
    scaler = Scaler(**meta[f"{role}_scaler"])
    return scaler, scaler.transform(frame.to_numpy(dtype=np.float32))


def predict_window(frame, meta, models, role, origin):
    """origin is the row immediately AFTER the final predictor observation."""
    l, h = meta["L"], meta["H"]
    if origin < l or origin > len(frame):
        raise ValueError("Forecast origin must follow a complete lookback window.")
    scaler, values = scale_frame(frame, meta, role)
    x = values[origin-l:origin][None]
    dates = frame.index[origin-l:origin].to_numpy()[None]
    p = {name: forecast_batches(models[name], x, dates=dates) for name in ("source_model", "adapted_model")}
    p["persistence"] = seasonal_naive(x, h, 1)
    p["seasonal_naive"] = seasonal_naive(x, h, meta["season"])
    p["input_mean"] = np.repeat(x.mean(1, keepdims=True), h, axis=1)
    truth = values[origin:origin+h][None] if origin+h <= len(frame) else None
    return scaler, x, truth, p


def score_forecasts(x, truth, predictions, meta, scaler):
    if truth is None:
        return pd.DataFrame()
    key = meta["columns"].index(meta["key"])
    rows = []
    for name, p in predictions.items():
        if p.shape != truth.shape:
            raise ValueError("Forecast and actual horizons must match exactly.")
        rows.append({"method": name, **all_metrics(x, truth, p, meta["season"], scaler, key)})
    return pd.DataFrame(rows).set_index("method")


def evaluate_history(frame, meta, models, role, maximum=1000):
    """Frozen-weight rolling evaluation; originals are never supplied to predictors."""
    l, h = meta["L"], meta["H"]
    origins = np.arange(l, len(frame)-h+1, h)
    origins = origins[-maximum:]
    if not len(origins):
        raise ValueError("No complete future is available for evaluation.")
    scaler, values = scale_frame(frame, meta, role)
    xs, ys, predictions = [], [], {name: [] for name in ("source_model", "adapted_model", "persistence", "seasonal_naive", "input_mean")}
    for start in range(0, len(origins), 128):
        ix = origins[start:start+128]
        x = np.stack([values[i-l:i] for i in ix])
        y = np.stack([values[i:i+h] for i in ix])
        dates = np.stack([frame.index[i-l:i].to_numpy() for i in ix])
        for name in ("source_model", "adapted_model"):
            predictions[name].append(forecast_batches(models[name], x, dates=dates))
        predictions["persistence"].append(seasonal_naive(x, h, 1))
        predictions["seasonal_naive"].append(seasonal_naive(x, h, meta["season"]))
        predictions["input_mean"].append(np.repeat(x.mean(1, keepdims=True), h, axis=1))
        xs.append(x);ys.append(y)
    x, truth = np.concatenate(xs), np.concatenate(ys)
    forecasts = {k: np.concatenate(v) for k, v in predictions.items()}
    scores = score_forecasts(x, truth, forecasts, meta, scaler)
    dates = np.concatenate([frame.index[i:i+h].to_numpy() for i in origins])
    export = pd.DataFrame({"date": np.repeat(dates, len(meta["columns"])), "variable": np.tile(meta["columns"], len(dates))})
    export["actual"] = np.concatenate([frame.iloc[i:i+h].to_numpy() for i in origins]).reshape(-1)
    for name, p in forecasts.items():
        export[name] = scaler.inverse(p).reshape(-1)
    return scores, export, origins


def benchmark_table(ds, horizon):
    if horizon == 1:
        return pd.read_csv(RESULTS_DIR / "forecast_horizons" / f"{ds}_next_hour_comparison.csv")
    family = "strict_context" if ds == "climate" else "strict_refined"
    f = pd.read_csv(RESULTS_DIR / family / ds / "metrics_all.csv")
    f = f[f.method.isin(["strict_source", "strict_adapted", "seasonal_naive", "persistence", "input_mean"])]
    f["family"] = "Strict forecasting hybrid / baselines"
    return f.groupby(["domain", "family", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index()
