"""LiteTCN-Forecaster: a lightweight (~0.2M params) multivariate forecaster built for adaptation.

  x (B,L,C) -> RevIN -> strided dilated Conv1d encoder -> latent z (B,d)   <- features aligned across domains
                                   |                        |-> forecast head  (B,H,C)
                                   |                        |-> reconstruction head (B,L,C)  <- self-supervised task
                 linear temporal path (DLinear-style, shared over channels) ---+-> forecast
"""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible instance normalisation (Kim et al., ICLR 2022) - removes per-window level/scale shift."""

    def __init__(self, C, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(C))
        self.b = nn.Parameter(torch.zeros(C))

    def norm(self, x):
        mu = x.mean(1, keepdim=True).detach()
        sd = (x.var(1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        return (x - mu) / sd * self.w + self.b, (mu, sd)

    def denorm(self, y, stats):
        mu, sd = stats
        return (y - self.b) / (self.w + self.eps) * sd + mu


class LiteTCNForecaster(nn.Module):
    def __init__(self, L, H, C, width=64, latent=128, dropout=0.1):
        super().__init__()
        self.L, self.H, self.C = L, H, C
        self.revin = RevIN(C)
        self.enc = nn.Sequential(
            nn.Conv1d(C, width // 2, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(width // 2, width, 5, stride=2, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=4, dilation=2), nn.GELU(),
        )
        with torch.no_grad():
            n = self.enc(torch.zeros(1, C, L)).numel()
        self.proj = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n, latent), nn.GELU())
        self.head = nn.Linear(latent, H * C)
        self.linear = nn.Linear(L, H)
        self.recon = nn.Linear(latent, L * C)

    def encode(self, xn):
        return self.proj(self.enc(xn.transpose(1, 2)))

    def encode_multi(self, xn):
        """Latent z (B,d) and time-pooled conv features (B,width) for multi-scale alignment."""
        h = self.enc(xn.transpose(1, 2))
        return self.proj(h), h.mean(-1)

    def forward(self, x, return_z=False, return_h=False):
        xn, stats = self.revin.norm(x)
        if return_h:
            z, h = self.encode_multi(xn)
        else:
            z = self.encode(xn)
            h = None
        yn = self.head(z).view(-1, self.H, self.C) + self.linear(xn.transpose(1, 2)).transpose(1, 2)
        y = self.revin.denorm(yn, stats)
        if return_h:
            return y, z, h
        return (y, z) if return_z else y

    def masked_recon_loss(self, x, ratio=0.3):
        """Self-supervised pretext: mask random time steps of the (label-free) input and reconstruct them."""
        xn, _ = self.revin.norm(x)
        mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) < ratio
        rec = self.recon(self.encode(xn.masked_fill(mask, 0.0))).view_as(xn)
        m = mask.expand_as(xn)
        return ((rec - xn.detach()) ** 2)[m].mean()

    def inner_forecast_loss(self, x):
        """Forecast-consistent pretext on unlabeled windows: predict the last H look-back
        steps from the preceding L-H steps (padded to length L). No future labels used."""
        L, H = self.L, self.H
        if L <= H:
            return x.new_zeros(())
        prefix = x[:, : L - H]
        pad = prefix[:, :1].expand(-1, H, -1)
        x_short = torch.cat([pad, prefix], 1)
        return nn.functional.mse_loss(self.forward(x_short), x[:, L - H:].detach())

    def forecast_consistency_loss(self, x, noise=0.1):
        """Two noisy views of the same unlabeled window should yield the same forecast."""
        scale = x.std(1, keepdim=True).clamp_min(1e-3)
        y1 = self.forward(x)
        y2 = self.forward(x + noise * torch.randn_like(x) * scale)
        return nn.functional.mse_loss(y1, y2)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
