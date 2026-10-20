"""Small ensemble with an explicitly calibrated weight for each forecast variable."""
import torch
from torch import nn

from .model import make_forecaster
from .refined_models import ChannelRidge


class HistoryEnsemble(nn.Module):
    def __init__(self, cfg, neural_spec):
        super().__init__()
        self.L, self.H, self.C = cfg["L"], cfg["H"], len(cfg["columns"])
        self.linear = ChannelRidge(self.L, self.H, self.C)
        self.neural = make_forecaster(cfg, neural_spec)
        self.dlinear = make_forecaster(cfg, {"architecture": "dlinear"})
        self.register_buffer("mix", torch.full((self.C, 5), .2))

    def components(self, x):
        return torch.stack((self.linear(x), self.neural(x), self.dlinear(x),
                            x.mean(1, keepdim=True).expand(-1, self.H, -1),
                            torch.zeros((len(x), self.H, self.C), device=x.device, dtype=x.dtype)), -1)

    def forward(self, x):
        return (self.components(x) * self.mix[None, None]).sum(-1)
