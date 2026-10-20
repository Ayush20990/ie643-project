"""Expose the first output of an existing direct multi-horizon forecaster."""
from torch import nn


class FirstHourReference(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base
        self.L,self.H,self.C=base.L,1,base.C
        self.requires_calendar=False

    def forward(self,x):
        return self.base(x[...,:self.C])[:,:1]
