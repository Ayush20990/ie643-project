"""Recent forecast-residual correction using observations inside one input window.

This is not learning from test future labels. At a forecast origin, replay a
forecast made 12 hours earlier, compare its first 12 outputs to the now observed
last 12 input values, then apply a decaying level/affine correction to the next
forecast. Windows are processed independently; no state crosses test origins.
"""
import torch
from torch import nn


def correct_prediction(prediction, past_prediction, past_observed, kind="level", strength=0., decay=24.):
    if strength == 0:
        return prediction
    steps = past_observed.shape[1]
    age = torch.arange(steps-1, -1, -1, dtype=prediction.dtype, device=prediction.device)
    weights = (-age/3.).exp() if kind == "innovation" else torch.ones_like(age)
    weights = weights/weights.sum()
    weights = weights[None, :, None]
    pm, ym = (past_prediction*weights).sum(1, keepdim=True), (past_observed*weights).sum(1, keepdim=True)
    if kind == "affine":
        covariance = ((past_prediction-pm)*(past_observed-ym)*weights).sum(1, keepdim=True)
        variance = ((past_prediction-pm).square()*weights).sum(1, keepdim=True)
        gain = ((covariance+.1)/(variance+.1)).clamp(.5, 1.5)
    else:
        gain = torch.ones_like(pm)
    offset = ym-gain*pm
    horizon = torch.arange(prediction.shape[1], dtype=prediction.dtype, device=prediction.device)
    attenuation = (-horizon/decay).exp()[None, :, None]
    return prediction + strength*attenuation*((gain-1)*prediction+offset)


class CalibratedLiteTCN(nn.Module):
    def __init__(self, base, kind="level", strength=0., decay=24., calibration_steps=12):
        super().__init__()
        self.base = base
        self.L, self.H, self.C = base.L, base.H, base.C
        self.requires_calendar = base.requires_calendar
        self.kind, self.strength, self.decay = kind, strength, decay
        self.calibration_steps = calibration_steps

    def forward(self, packed):
        prediction = self.base(packed)
        if self.strength == 0:
            return prediction
        prefix = packed[:, :-self.calibration_steps]
        past_prediction = self.base(prefix)[:, :self.calibration_steps]
        past_observed = packed[:, -self.calibration_steps:, :self.C]
        return correct_prediction(prediction, past_prediction, past_observed,
                                  self.kind, self.strength, self.decay)
