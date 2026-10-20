"""Causal regime-dependent mixture of source/adapted next-hour experts."""
import torch
from torch import nn

from .context_models import ContextEnsemble
from .nowcast_hybrid import NextHourHybrid
from .strict_models import StrictEnsemble


def observed_regime(packed,channels):
    x=packed[:,-48:,:channels]
    current=x[:,-1]-x[:,-2]
    previous=x[:,-2]-x[:,-3]
    acceleration=current-previous
    tau=torch.where(acceleration.abs()>1e-6,-current/acceleration,torch.full_like(current,float('inf')))
    approaching=(tau>=0)&(tau<=3)
    regime=torch.zeros_like(current,dtype=torch.long)
    regime[(current>0)&(acceleration<0)&approaching]=1
    regime[(current<0)&(acceleration>0)&approaching]=2
    flat=(current.abs()<.05*x.std(1,unbiased=False).clamp_min(.05))|(current*previous<=0)
    regime[flat]=3
    return regime


def episode_prefix(packed,end,channels):
    if end<48 or end>=packed.shape[1]:raise ValueError("Invalid within-input origin")
    prefix=packed[:,:end]
    pad=prefix[:,:1].expand(-1,packed.shape[1]-end,-1)
    return torch.cat((pad,prefix),1),packed[:,end:end+1,:channels].detach()


class PeakNowcast(nn.Module):
    def __init__(self,cfg,reference_kind,neural_spec):
        super().__init__()
        self.L,self.H,self.C=cfg["L"],1,len(cfg["columns"])
        self.requires_calendar=True
        self.current=NextHourHybrid(cfg)
        ref_cfg={**cfg,"H":24}
        self.reference=(ContextEnsemble if reference_kind=="context" else StrictEnsemble)(ref_cfg,neural_spec)
        self.register_buffer("current_source_mix",torch.zeros(self.C,5));self.current_source_mix[:,0]=1.
        self.register_buffer("current_adapted_mix",torch.zeros(self.C,5));self.current_adapted_mix[:,1]=1.
        self.register_buffer("reference_source_mix",torch.zeros(self.C,8));self.reference_source_mix[:,2]=1.
        self.register_buffer("reference_adapted_mix",torch.zeros(self.C,8));self.reference_adapted_mix[:,3]=1.
        self.register_buffer("mix",torch.zeros(self.C,4,5));self.mix[:,:,0]=1.

    def experts(self,x):
        current=self.current.components(x)
        reference=self.reference.components(x[...,:self.C])[:,:1]
        p=[(current*self.current_source_mix[None,None]).sum(-1),
           (current*self.current_adapted_mix[None,None]).sum(-1),
           (reference*self.reference_source_mix[None,None]).sum(-1),
           (reference*self.reference_adapted_mix[None,None]).sum(-1),x[:,-1:,:self.C]]
        return torch.stack(p,-1)

    def forward(self,x):
        regime=observed_regime(x,self.C)
        weights=self.mix[torch.arange(self.C,device=x.device)[None],regime]
        return (self.experts(x)*weights[:,None]).sum(-1)

    @torch.no_grad()
    def remove_inactive_target_weights(self):
        """Source-only serialization, with all adapted expert routes inactive."""
        self.current.models["tcn_adapted"].load_state_dict(self.current.models["tcn_source"].state_dict())
        self.current.models["dynamics_adapted"].load_state_dict(self.current.models["dynamics_source"].state_dict())
        self.current_adapted_mix.copy_(self.current_source_mix)
        self.current.mix.copy_(self.current_source_mix)
        for branch in ("linear","neural","dlinear"):
            self.reference.models[f"{branch}_adapted"].load_state_dict(self.reference.models[f"{branch}_source"].state_dict())
        self.reference_adapted_mix.copy_(self.reference_source_mix)
        self.reference.mix.copy_(self.reference_source_mix)
