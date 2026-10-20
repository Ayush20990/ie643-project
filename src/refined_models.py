"""Channel-specific regularized forecaster and a compact neural/linear ensemble.

This is a project-specific hybrid, not an implementation of a named paper.
All feature transforms use only the supplied look-back window.
"""
import numpy as np
import torch
from torch import nn

from .model import make_forecaster

NORMALIZATIONS = ("none", "mean", "last", "revin")


def channel_features(x, mode):
    """Return features [B,C,F], output offset and output scale [B,C]."""
    mean = x.mean(1)
    std = x.std(1, unbiased=False).clamp_min(.05)
    center = torch.zeros_like(mean) if mode == "none" else x[:, -1] if mode == "last" else mean
    scale = std if mode == "revin" else torch.ones_like(std)
    z = (x - center[:, None]) / scale[:, None]
    # Recent cross-variable levels, plus own level/volatility. Preserve statistics
    # removed by normalization so a stationary return can shrink towards zero.
    widths = sorted(set((1, min(6, x.shape[1]), min(24, x.shape[1]), x.shape[1])))
    cross = torch.cat([z[:, -w:].mean(1) for w in widths], -1)
    own = torch.stack((mean, std.log(), x[:, -1]), -1)
    ones = torch.ones((*mean.shape, 1), dtype=x.dtype, device=x.device)
    f = torch.cat((z.transpose(1, 2), cross[:, None].expand(-1, x.shape[2], -1), own, ones), -1)
    return f, center, scale


class ChannelRidge(nn.Module):
    def __init__(self, L, H, C):
        super().__init__()
        self.L, self.H, self.C = L, H, C
        F = channel_features(torch.zeros(1, L, C), "none")[0].shape[-1]
        self.register_buffer("weight", torch.zeros(C, F, H))
        self.register_buffer("mode", torch.zeros(C, dtype=torch.long))

    def forward(self, x):
        result = torch.empty((len(x), self.H, self.C), device=x.device, dtype=x.dtype)
        for i, mode in enumerate(NORMALIZATIONS):
            channels = self.mode == i
            if channels.any():
                f, offset, scale = channel_features(x, mode)
                y = torch.einsum("bcf,cfh->bch", f[:, channels], self.weight[channels])
                result[:, :, channels] = (y * scale[:, channels, None] + offset[:, channels, None]).transpose(1, 2)
        return result


class RefinedForecaster(nn.Module):
    def __init__(self, cfg, neural_spec):
        super().__init__()
        self.L, self.H, self.C = cfg["L"], cfg["H"], len(cfg["columns"])
        self.linear = ChannelRidge(self.L, self.H, self.C)
        self.neural = make_forecaster(cfg, neural_spec)
        # Convex weights per variable: linear, neural, input mean, training mean.
        self.register_buffer("mix", torch.zeros(self.C, 4))
        self.mix[:, 1] = 1

    def forward(self, x):
        components = torch.stack((self.linear(x), self.neural(x),
                                  x.mean(1, keepdim=True).expand(-1, self.H, -1),
                                  torch.zeros((len(x), self.H, self.C), device=x.device, dtype=x.dtype)), -1)
        return (components * self.mix[None, None]).sum(-1)


@torch.no_grad()
def sufficient_statistics(x, y, mode, batch_size=512):
    """Exact normal equations in forecast units, with a small feature ridge.

    Weight each row by output scale: minimizing this system minimizes the
    original standardized forecast MSE, even for window-normalized features.
    """
    gram = rhs = None
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start + batch_size], y[start:start + batch_size]
        f, offset, scale = channel_features(xb, mode)
        a = (f * scale[:, :, None]).double().transpose(0, 1)
        b = (yb - offset[:, None]).double().permute(2, 0, 1)
        g, r = a.transpose(1, 2) @ a, a.transpose(1, 2) @ b
        gram, rhs = (g, r) if gram is None else (gram + g, rhs + r)
    return gram / len(x), rhs / len(x)


def ridge_path(gram, rhs, penalties, reference=None):
    """Minimize MSE + penalty * ||W-reference||^2; intercept also anchored."""
    values, vectors = torch.linalg.eigh(gram)
    values = values.clamp_min(0)
    for penalty in penalties:
        target = rhs if reference is None else rhs + penalty * reference.double()
        projected = vectors.transpose(1, 2) @ target
        yield penalty, (vectors @ (projected / (values[:, :, None] + penalty))).float()


def convex_weights(predictions, truth, shrink=.001):
    """Fit a simplex mixture on validation only, with a weak uniform prior."""
    from scipy.optimize import minimize
    weights = []
    for channel in range(truth.shape[-1]):
        p = np.stack([a[..., channel].reshape(-1) for a in predictions], -1).astype(np.float64)
        y = truth[..., channel].reshape(-1).astype(np.float64)
        gram, rhs = p.T @ p / len(y), p.T @ y / len(y)
        k = len(predictions)
        # Scale penalty to the response variance, including low-variance series.
        lam = shrink * max(float(np.var(y)), 1e-6)
        objective = lambda w: float(w @ gram @ w - 2 * w @ rhs + lam * np.sum((w - 1 / k) ** 2))
        jacobian = lambda w: 2 * (gram @ w - rhs + lam * (w - 1 / k))
        result = minimize(objective, np.full(k, 1 / k), jac=jacobian,
                          bounds=[(0, 1)] * k, constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                          "jac": lambda w: np.ones_like(w)}, method="SLSQP", options={"ftol": 1e-12, "maxiter": 300})
        if not result.success:
            raise RuntimeError(f"Convex mixture fit failed: {result.message}")
        w = np.maximum(result.x, 0)
        weights.append(w / w.sum())
    return np.asarray(weights, dtype=np.float32)
