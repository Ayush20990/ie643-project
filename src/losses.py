"""Domain-adaptation losses."""
import torch
import torch.nn as nn


def mmd_rbf(a, b, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Multi-kernel MMD^2 (Gretton et al. 2012; Long et al. 2015), bandwidth from the median heuristic."""
    x = torch.cat([a, b], 0)
    d2 = torch.cdist(x, x).pow(2)
    bw = d2.detach().median().clamp_min(1e-6)
    k = sum(torch.exp(-d2 / (bw * s)) for s in scales)
    n = a.shape[0]
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


def coral(a, b):
    """Deep CORAL (Sun & Saenko 2016): distance between feature covariances."""
    d = a.shape[1]

    def cov(x):
        x = x - x.mean(0, keepdim=True)
        return x.T @ x / (x.shape[0] - 1)

    return (cov(a) - cov(b)).pow(2).sum() / (4 * d * d)


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DomainDiscriminator(nn.Module):
    """DANN (Ganin et al. 2016): domain classifier trained through a gradient-reversal layer."""

    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z, lam=1.0):
        return self.net(_GRL.apply(z, lam)).squeeze(-1)


def l2_sp(model, ref):
    """L2-SP anchor (Li et al. 2018): keep adapted weights close to the source solution (anti-forgetting)."""
    return sum(((p - ref[n]) ** 2).sum() for n, p in model.named_parameters() if n in ref)


def spectral_align(xs, xt):
    """RAINCOAT-style frequency alignment: match mean amplitude spectra of the two domains."""
    as_ = torch.fft.rfft(xs, dim=1).abs().mean(0)
    at = torch.fft.rfft(xt, dim=1).abs().mean(0)
    return (as_ - at).pow(2).mean()
