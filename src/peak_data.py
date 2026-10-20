"""Pack only past values and deterministic calendar covariates, preserving old splits."""
import numpy as np
import torch

from .config import DATASETS
from .data import load_domain, prepare
from .peak_model import calendar_features


def prepare_peak(ds):
    data = prepare(ds)
    cfg = DATASETS[ds]
    if ds not in ("climate", "energy"):
        raise ValueError("Peak experiments currently cover the hourly climate and energy datasets")
    for role in ("source", "target"):
        dates = load_domain(ds, cfg[role]).sort_values("date").date
        a, b = int(len(dates) * cfg["split"][0]), int(len(dates) * sum(cfg["split"][:2]))
        slices = {"train": dates.iloc[:a], "val": dates.iloc[a-cfg["L"]:b], "test": dates.iloc[b-cfg["L"]:]}
        for split, times in slices.items():
            clock = calendar_features(times)
            stride = cfg["train_stride"] if split == "train" else 1
            windows = np.lib.stride_tricks.sliding_window_view(clock, cfg["L"], axis=0).transpose(0, 2, 1)[::stride]
            x, y = data[role][split]
            windows = torch.from_numpy(np.array(windows[:len(x)], copy=True))
            data[role][f"{split}_packed"] = (torch.cat((x, windows), -1), y)
    return data
