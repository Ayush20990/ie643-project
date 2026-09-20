"""Forecast-quality and adaptation-quality metrics."""
import numpy as np


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def mase(x, y, p, season):
    """MAE of forecast / in-sample MAE of the seasonal-naive forecast on each input window (scale-free)."""
    naive_err = np.mean(np.abs(x[:, season:] - x[:, :-season]), axis=(1, 2)) + 1e-8
    return float(np.mean(np.mean(np.abs(y - p), axis=(1, 2)) / naive_err))


def seasonal_naive(x, H, season):
    reps = int(np.ceil(H / season))
    return np.tile(x[:, -season:], (1, reps, 1))[:, :H]


def all_metrics(x, y, p, season, scaler=None, key_idx=None):
    """x,y,p in scaled space (N,L|H,C). Optionally RMSE/MAE of the key variable in original units."""
    out = {"MSE": mse(y, p), "MAE": mae(y, p), "MASE": mase(x, y, p, season)}
    if scaler is not None and key_idx is not None:
        yk = y[..., key_idx] * scaler.std[key_idx] + scaler.mean[key_idx]
        pk = p[..., key_idx] * scaler.std[key_idx] + scaler.mean[key_idx]
        out["key_RMSE"] = float(np.sqrt(np.mean((yk - pk) ** 2)))
        out["key_MAE"] = float(np.mean(np.abs(yk - pk)))
    return out


def gap_closure(err_src_only, err_method, err_oracle):
    """Fraction of the source-only -> oracle (target fine-tuned) error gap closed without target labels.
    1 = as good as the supervised oracle, 0 = no gain, <0 = negative transfer."""
    gap = err_src_only - err_oracle
    return float((err_src_only - err_method) / gap) if abs(gap) > 1e-12 else float("nan")
