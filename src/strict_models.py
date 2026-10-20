"""Source/adapter ensemble used by the strict input-window-only experiment."""
import torch
from torch import nn

from .model import make_forecaster
from .refined_models import ChannelRidge

BRANCHES = ("linear_source", "linear_adapted", "neural_source", "neural_adapted", "dlinear_source", "dlinear_adapted")
COMPONENTS = BRANCHES + ("input_mean", "training_mean")


def input_pretext(inputs, horizon):
    """Withhold H values INSIDE each input; no cross-window reconstruction.

    Left padding exactly matches the previous experiment's inner-forecast task.
    The forecast origin precedes the end of this input; target Y is never used.
    """
    if horizon < 1 or horizon >= inputs.shape[1]:
        raise ValueError("Pretext horizon must be smaller than input length")
    prefix = inputs[:, :-horizon]
    context = torch.cat((prefix[:, :1].expand(-1, horizon, -1), prefix), 1)
    return context, inputs[:, -horizon:].detach()


class StrictEnsemble(nn.Module):
    def __init__(self, cfg, neural_spec):
        super().__init__()
        self.L, self.H, self.C = cfg["L"], cfg["H"], len(cfg["columns"])
        self.models = nn.ModuleDict()
        for name in BRANCHES:
            self.models[name] = (ChannelRidge(self.L, self.H, self.C) if name.startswith("linear") else
                                 make_forecaster(cfg, {"architecture": "dlinear"} if name.startswith("dlinear") else neural_spec))
        self.register_buffer("mix", torch.zeros(self.C, len(COMPONENTS)))
        self.mix[:, 2] = 1

    def components(self, x):
        predictions = [self.models[name](x) for name in BRANCHES]
        predictions.extend((x.mean(1, keepdim=True).expand(-1, self.H, -1),
                            torch.zeros((len(x), self.H, self.C), dtype=x.dtype, device=x.device)))
        return torch.stack(predictions, -1)

    def forward(self, x):
        y = torch.zeros((len(x), self.H, self.C), dtype=x.dtype, device=x.device)
        for i, name in enumerate(BRANCHES):
            if bool((self.mix[:, i] > 1e-7).any()):
                y = y + self.models[name](x) * self.mix[None, :, i][:, None]
        return y + x.mean(1, keepdim=True) * self.mix[None, :, 6][:, None]
