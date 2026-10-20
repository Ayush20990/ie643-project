"""LiteTCN revision for shape-sensitive forecasting; original models are preserved.

Retains a lightweight dilated convolutional encoder and direct multi-horizon head.
Changes: seasonal folding that preserves hourly positions, channel-specific linear
skip, future-known calendar features, and attention over observed history. This is a project model,
not a reproduction of TiDE, TFT or DILATE.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

TIME_PERIODS = (24., 168., 365.2425 * 24.)


def calendar_features(dates):
    dates = pd.DatetimeIndex(np.asarray(dates).reshape(-1)).as_unit("ns")
    if dates.hasnans:
        raise ValueError("Calendar-aware forecasting requires valid dates")
    # Local timestamps are kept in the same wall-clock convention as training.
    hours = dates.asi8.astype(np.float64) / 3.6e12
    return np.stack([f(2 * np.pi * hours / period) for period in TIME_PERIODS for f in (np.sin, np.cos)], -1).astype(np.float32)


def pack_calendar(values, dates):
    values = np.asarray(values, dtype=np.float32)
    dates = np.asarray(dates)
    if dates.shape != values.shape[:2]:
        raise ValueError("One timestamp is required for every input row")
    features = calendar_features(dates).reshape(*dates.shape, 6)
    return np.concatenate((values, features), -1)


class CausalResidualBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.conv1 = nn.Conv1d(width, width, 3, dilation=dilation, padding=self.padding, groups=width)
        self.conv2 = nn.Conv1d(width, width, 1)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        length = x.shape[-1]
        h = self.conv1(x)[..., :length]
        h = self.dropout(nn.functional.gelu(self.norm1(h.transpose(1, 2)))).transpose(1, 2)
        h = self.conv2(h)[..., :length]
        h = self.dropout(nn.functional.gelu(self.norm2(h.transpose(1, 2)))).transpose(1, 2)
        return x + h


class PeakLiteTCN(nn.Module):
    def __init__(self, cfg, calendar=True, width=32, dropout=.1, encoder="tcn"):
        super().__init__()
        self.L, self.H, self.C = cfg["L"], cfg["H"], len(cfg["columns"])
        self.context = self.L - self.H
        self.season = cfg["season"]
        if self.context % self.season:
            raise ValueError("Seasonal folding requires complete cycles in the effective context")
        self.cycles = self.context // self.season
        self.requires_calendar, self.encoder_type = calendar, encoder
        self.input_projection = nn.Linear(self.cycles * self.C + 6, width)
        self.position = nn.Parameter(torch.zeros(1, self.season, width))
        if encoder == "tcn":
            self.encoder = nn.Sequential(*(CausalResidualBlock(width, d, dropout) for d in (1, 2, 4, 8)))
            self.query = nn.Linear(6, width)
            self.horizon = nn.Parameter(torch.randn(1, self.H, width) * .02)
            self.attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        elif encoder == "dense":
            # Optional TiDE-inspired dense comparator, using identical inputs/skip/head.
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(self.context * (self.C + 6) + 2 * self.C, 128),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 128), nn.GELU(),
                nn.Linear(128, self.H * width))
        else:
            raise ValueError(encoder)
        self.temporal_decoder = nn.Sequential(nn.Linear(2 * width + 5 * self.C + 6, 64), nn.GELU(),
                                               nn.Dropout(dropout), nn.Linear(64, self.C))
        nn.init.zeros_(self.temporal_decoder[-1].weight)
        nn.init.zeros_(self.temporal_decoder[-1].bias)
        self.linear = nn.Parameter(torch.zeros(self.C, self.H, self.context))
        self.linear_bias = nn.Parameter(torch.zeros(self.H, self.C))
        with torch.no_grad():
            for h in range(self.H):
                self.linear[:, h, self.context - self.season + h % self.season] = 1.

    def future_clock(self, last):
        out = []
        step = torch.arange(1, self.H + 1, dtype=last.dtype, device=last.device)[None]
        for i, period in enumerate(TIME_PERIODS):
            sin, cos = last[:, 2*i:2*i+1], last[:, 2*i+1:2*i+2]
            angle = step * (2 * torch.pi / period)
            out.extend((sin * angle.cos() + cos * angle.sin(), cos * angle.cos() - sin * angle.sin()))
        return torch.stack(out, -1)

    def forward(self, packed):
        if packed.shape[1] < self.context:
            raise ValueError(f"Need {self.context} real context steps")
        x = packed[:, -self.context:, :self.C]
        if packed.shape[-1] == self.C + 6:
            time = packed[:, -self.context:, self.C:]
        elif self.requires_calendar:
            raise ValueError("This model requires timestamp-derived calendar features")
        else:
            time = torch.zeros((*x.shape[:2], 6), dtype=x.dtype, device=x.device)
        if not self.requires_calendar:
            time = torch.zeros_like(time)
        mean = x.mean(1, keepdim=True).detach()
        sd = x.std(1, keepdim=True, unbiased=False).clamp_min(.05).detach()
        z = (x - mean) / sd
        stats = torch.cat((mean[:, 0], sd[:, 0].log()), -1)
        features = torch.cat((z, time), -1)
        # One token per hourly position, with previous days stacked as features.
        # Every observed value remains represented; no temporal averaging/stride.
        folded = z.reshape(len(x), self.cycles, self.season, self.C).permute(0, 2, 1, 3).reshape(len(x), self.season, -1)
        folded = torch.cat((folded, time[:, -self.season:]), -1)
        embedded = self.input_projection(folded) + self.position
        future = self.future_clock(time[:, -1]) if self.requires_calendar else time[:, :self.H]
        if self.encoder_type == "tcn":
            memory = self.encoder(embedded.transpose(1, 2)).transpose(1, 2)
            query = self.query(future) + self.horizon
            context = self.attention(query, memory, memory, need_weights=False)[0]
            last = memory[:, -1:]
        else:
            flat = torch.cat((features.flatten(1), stats), -1)
            context = self.encoder(flat).view(len(x), self.H, -1)
            last = embedded[:, -1:]
        seasonal = torch.stack([z[:, -self.season + h % self.season] for h in range(self.H)], 1)
        values = torch.cat((z[:, -1], z[:, -1] - z[:, -2], stats), -1)
        decoded = self.temporal_decoder(torch.cat((context, last.expand(-1, self.H, -1), seasonal,
                                                   values[:, None].expand(-1, self.H, -1), future), -1))
        linear = torch.einsum("blc,chl->bhc", z, self.linear) + self.linear_bias
        return (linear + decoded) * sd + mean


def forecast_loss(prediction, truth, key, shape=False):
    weights = prediction.new_ones(prediction.shape[-1])
    weights[key] = 3.
    point = (((prediction - truth) ** 2) * weights).mean() / weights.mean()
    if not shape:
        return point
    ramp = (((prediction.diff(dim=1) - truth.diff(dim=1)) ** 2) * weights).mean() / weights.mean()
    extrema = (((prediction.amax(1) - truth.amax(1)) ** 2 +
                (prediction.amin(1) - truth.amin(1)) ** 2) * weights).mean() / weights.mean()
    return point + .25 * ramp + .1 * extrema


def pretext(packed, horizon, channels):
    return packed[:, :-horizon], packed[:, -horizon:, :channels].detach()
