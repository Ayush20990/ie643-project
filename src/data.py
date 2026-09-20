"""Loading, per-domain scaling, chronological splitting and windowing."""
import json

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, PROC_DIR


class Scaler:
    """Per-domain z-score. Target-domain statistics only use unlabeled target *inputs* (train split)."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, arrays):
        x = np.concatenate(arrays, 0)
        return cls(x.mean(0), x.std(0) + 1e-6)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def load_domain(ds, domain):
    return pd.read_csv(PROC_DIR / ds / f"{domain}.csv", parse_dates=["date"])


def series_list(df, cols):
    if "series" in df.columns:
        return [g.sort_values("date")[cols].values.astype(np.float32) for _, g in df.groupby("series", sort=False)]
    return [df.sort_values("date")[cols].values.astype(np.float32)]


def chrono_split(arrays, split, L):
    """Chronological split per series. val/test segments are prefixed with L steps of context."""
    out = {"train": [], "val": [], "test": []}
    for x in arrays:
        n = len(x)
        a, b = int(n * split[0]), int(n * (split[0] + split[1]))
        out["train"].append(x[:a])
        out["val"].append(x[a - L:b])
        out["test"].append(x[b - L:])
    return out


def make_windows(arrays, L, H, stride=1):
    xs, ys = [], []
    for x in arrays:
        if len(x) < L + H:
            continue
        w = np.lib.stride_tricks.sliding_window_view(x, L + H, axis=0)[::stride]  # (N, C, L+H)
        w = np.ascontiguousarray(w.transpose(0, 2, 1))
        xs.append(w[:, :L])
        ys.append(w[:, L:])
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))


def prepare(ds):
    """Returns dict with scaled windows for both domains + scalers."""
    cfg = DATASETS[ds]
    L, H, cols = cfg["L"], cfg["H"], cfg["columns"]
    out = {"cfg": cfg}
    for role in ["source", "target"]:
        arrays = series_list(load_domain(ds, cfg[role]), cols)
        parts = chrono_split(arrays, cfg["split"], L)
        scaler = Scaler.fit(parts["train"])
        d = {"scaler": scaler}
        for s in ["train", "val", "test"]:
            scaled = [scaler.transform(a) for a in parts[s]]
            stride = cfg["train_stride"] if s == "train" else 1
            d[s] = make_windows(scaled, L, H, stride)
        out[role] = d
    return out


def batches(X, Y=None, bs=256, shuffle=True, gen=None):
    n = len(X)
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield (X[j], Y[j]) if Y is not None else X[j]


def infinite(X, Y=None, bs=256, gen=None):
    while True:
        yield from batches(X, Y, bs, True, gen)


def save_meta(ds, data, path):
    cfg = data["cfg"]
    meta = {k: v for k, v in cfg.items()}
    meta["source_scaler"] = data["source"]["scaler"].to_dict()
    meta["target_scaler"] = data["target"]["scaler"].to_dict()
    path.write_text(json.dumps(meta, indent=2))
