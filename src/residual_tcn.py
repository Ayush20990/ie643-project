"""Original LiteTCN encoder with a strong linear skip and regularized residual head.

The DLinear source forecast is folded exactly into the shared linear skip. The
original three strided convolutions and source-trained projection are retained.
Only the small neural forecast head is refitted/adapted; zero residual is an
explicit baseline. The unused reconstruction head is removed.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .model import LiteTCNForecaster


class ResidualLiteTCN(LiteTCNForecaster):
    def __init__(self, cfg):
        super().__init__(cfg["L"], cfg["H"], len(cfg["columns"]))
        del self.recon
        self.requires_calendar = False
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("head."))

    @torch.no_grad()
    def initialize(self, source_tcn, source_dlinear):
        self.enc.load_state_dict(source_tcn.enc.state_dict())
        self.proj.load_state_dict(source_tcn.proj.state_dict())
        self.revin.load_state_dict(source_dlinear.revin.state_dict())
        basis = torch.eye(self.L)
        trend = F.avg_pool1d(F.pad(basis[:, None], (source_dlinear.kernel//2,)*2, mode="replicate"),
                             source_dlinear.kernel, stride=1)[:, 0]
        bias = source_dlinear.head.bias + source_dlinear.trend.bias
        response = source_dlinear.head(basis-trend) + source_dlinear.trend(trend) - bias
        self.linear.weight.copy_(response.T)
        self.linear.bias.copy_(bias)
        self.head.weight.zero_()
        self.head.bias.zero_()
        return self.eval()

    def head_weights(self):
        weight = self.head.weight.reshape(self.H, self.C, -1).permute(1, 2, 0)
        bias = self.head.bias.reshape(self.H, self.C).T[:, None]
        return torch.cat((weight, bias), 1)

    @torch.no_grad()
    def set_head(self, weight):
        self.head.weight.copy_(weight[:, :-1].permute(2, 0, 1).reshape(self.H*self.C, -1))
        self.head.bias.copy_(weight[:, -1].T.reshape(-1))

    def baseline_and_design(self, x):
        normalized, stats = self.revin.norm(x)
        latent = self.encode(normalized)
        baseline = self.revin.denorm(self.linear(normalized.transpose(1, 2)).transpose(1, 2), stats)
        features = torch.cat((latent, torch.ones_like(latent[:, :1])), -1)
        scale = stats[1][:, 0]/(self.revin.w+self.revin.eps)
        return baseline, features[:, None]*scale[:, :, None]
