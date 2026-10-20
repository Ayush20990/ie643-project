"""Small next-hour TCN plus source-anchored local multivariate dynamics."""
import torch
from torch import nn

from .nowcast_tcn import NextHourTCN


def dynamics_features(packed,channels,context=48):
    x=packed[:,-context:,:channels]
    last=x[:,-1]
    sd=x.std(1,unbiased=False).clamp_min(.05)
    z=(x-last[:,None])/sd[:,None]
    cross=torch.cat([z[:,-w:].mean(1) for w in (1,3,6,12,24,context)],-1)
    cross=torch.cat((cross,(x[:,-1]-x[:,-2])/sd,packed[:,-1,channels:]),-1)
    own=torch.stack((last,x.mean(1),sd.log()),-1)
    ones=torch.ones_like(last[...,None])
    f=torch.cat((z.transpose(1,2),cross[:,None].expand(-1,channels,-1),own,ones),-1)
    # Fit in original standardized forecast units, with an observed-only scale.
    return f*sd[:,:,None],last


class LocalDynamics(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.L,self.H,self.C=cfg["L"],1,len(cfg["columns"])
        self.requires_calendar=True
        f,_=dynamics_features(torch.zeros(1,self.L,self.C+6),self.C)
        self.register_buffer("weight",torch.zeros(self.C,f.shape[-1],1))

    def forward(self,x):
        f,last=dynamics_features(x,self.C)
        return last[:,None]+torch.einsum("bcf,cfh->bhc",f,self.weight)


class NextHourHybrid(nn.Module):
    names=("tcn_source","tcn_adapted","dynamics_source","dynamics_adapted")
    def __init__(self,cfg):
        super().__init__()
        self.L,self.H,self.C=cfg["L"],1,len(cfg["columns"])
        self.requires_calendar=True
        self.models=nn.ModuleDict({name:NextHourTCN(cfg) if name.startswith("tcn") else LocalDynamics(cfg) for name in self.names})
        self.register_buffer("mix",torch.zeros(self.C,5));self.mix[:,0]=1.

    def components(self,x):
        p=[self.models[n](x) for n in self.names]
        p.append(x[:,-1:,:self.C])
        return torch.stack(p,-1)

    def forward(self,x):
        p=torch.zeros((len(x),1,self.C),device=x.device,dtype=x.dtype)
        for i,name in enumerate(self.names):
            if bool((self.mix[:,i]>1e-7).any()):p=p+self.models[name](x)*self.mix[:,i][None,None]
        return p+x[:,-1:,:self.C]*self.mix[:,4][None,None]
