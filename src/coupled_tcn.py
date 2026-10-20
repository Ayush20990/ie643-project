"""Small temperature correction from LiteTCN-predicted drivers, never true futures.

The frozen LiteTCN still forecasts every channel. This head couples those joint
forecasts to a temperature residual. It is not a physical heat-balance model or
a reproduction of a published architecture.
"""
import torch
from torch import nn

from .peak_model import PeakLiteTCN, TIME_PERIODS
from .residual_tcn import ResidualLiteTCN


class CoupledLiteTCN(nn.Module):
    def __init__(self, cfg, base_kind, base_spec=None, control="coupled"):
        super().__init__()
        self.L, self.H, self.C = cfg["L"], cfg["H"], len(cfg["columns"])
        self.key = cfg["columns"].index(cfg["key"])
        self.base_kind = base_kind
        self.base = PeakLiteTCN(cfg, **(base_spec or {})) if base_kind=="peak" else ResidualLiteTCN(cfg)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.requires_calendar = True
        self.decoder = nn.Sequential(nn.Linear(6*self.C+7,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.register_buffer("feature_mask",torch.ones(6*self.C+7))
        self.set_control(control)

    def set_control(self, control):
        if control not in ("temperature_only","coupled"):
            raise ValueError(control)
        self.feature_mask.fill_(1.)
        if control=="temperature_only":
            for j in range(6):
                self.feature_mask[j*self.C:(j+1)*self.C] = 0.
                self.feature_mask[j*self.C+self.key] = 1.

    def future_clock(self, last):
        out=[]
        step=torch.arange(1,self.H+1,dtype=last.dtype,device=last.device)[None]
        for i,period in enumerate(TIME_PERIODS):
            sin,cos=last[:,2*i:2*i+1],last[:,2*i+1:2*i+2]
            angle=step*(2*torch.pi/period)
            out.extend((sin*angle.cos()+cos*angle.sin(),cos*angle.cos()-sin*angle.sin()))
        return torch.stack(out,-1)

    @torch.no_grad()
    def cache_features(self, packed):
        if packed.shape[-1] != self.C+6:
            raise ValueError("Coupled LiteTCN requires timestamp features")
        self.base.eval()
        x=packed[...,:self.C]
        base=self.base(packed if self.base_kind=="peak" else x)
        mean=x.mean(1,keepdim=True)
        sd=x.std(1,keepdim=True,unbiased=False).clamp_min(.05)
        z=(x-mean)/sd
        p=(base-mean)/sd
        last=z[:,-1:]
        stats=torch.cat((mean,sd.log(),last,z[:,-6:].mean(1,keepdim=True)),-1).expand(-1,self.H,-1)
        clock=self.future_clock(packed[:,-1,self.C:])
        horizon=torch.arange(1,self.H+1,dtype=x.dtype,device=x.device)[None,:,None].expand(len(x),-1,-1)/self.H
        features=torch.cat((p,p-last,stats,clock,horizon),-1)
        return features,base,sd[...,self.key:self.key+1].expand(-1,self.H,-1)

    def from_features(self, features, base, scale):
        correction=self.decoder(features*self.feature_mask)*scale
        onehot=nn.functional.one_hot(torch.tensor(self.key,device=base.device),self.C).to(base.dtype)
        return base+correction*onehot

    def forward(self, packed):
        return self.from_features(*self.cache_features(packed))
