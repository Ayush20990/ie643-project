"""Small unstrided TCN for genuine next-hour multivariate forecasting."""
import torch
from torch import nn

from .peak_model import CausalResidualBlock, TIME_PERIODS


class NextHourTCN(nn.Module):
    def __init__(self,cfg,width=32,context=48):
        super().__init__()
        self.L,self.H,self.C=cfg["L"],1,len(cfg["columns"])
        self.context=context
        self.requires_calendar=True
        self.projection=nn.Conv1d(2*self.C+6,width,1)
        self.encoder=nn.Sequential(*(CausalResidualBlock(width,d,.05) for d in (1,2,4)))
        self.head=nn.Sequential(nn.Linear(width+3*self.C+6,64),nn.GELU(),nn.Linear(64,self.C))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,packed):
        if packed.shape[-1]!=self.C+6 or packed.shape[1]<self.context:
            raise ValueError("Need real recent history and timestamp features")
        x=packed[:,-self.context:,:self.C]
        clock=packed[:,-self.context:,self.C:]
        last=x[:,-1:]
        sd=x.std(1,keepdim=True,unbiased=False).clamp_min(.05).detach()
        z=(x-last)/sd
        difference=torch.cat((torch.zeros_like(z[:,:1]),z.diff(dim=1)),1)
        h=self.encoder(self.projection(torch.cat((z,difference,clock),-1).transpose(1,2)))[:,:,-1]
        future=[]
        for j,period in enumerate(TIME_PERIODS):
            angle=2*torch.pi/period
            sin,cos=clock[:,-1,2*j],clock[:,-1,2*j+1]
            future.extend((sin*torch.cos(x.new_tensor(angle))+cos*torch.sin(x.new_tensor(angle)),
                           cos*torch.cos(x.new_tensor(angle))-sin*torch.sin(x.new_tensor(angle))))
        inputs=torch.cat((h,difference[:,-1],last[:,0],sd[:,0].log(),torch.stack(future,-1)),-1)
        return last+sd*self.head(inputs)[:,None]


def within_input(packed,channels):
    """Last observed hour is the pretext target, never part of its predictor."""
    return packed[:,:-1],packed[:,-1:,:channels].detach()


def turning_weights(prefix,truth,key):
    before=prefix[:,-1,key]-prefix[:,-2,key]
    after=truth[:,0,key]-prefix[:,-1,key]
    return 1.+2.*((before*after<0)&(before.abs()>.01)&(after.abs()>.01)).to(prefix.dtype)


def forecast_loss(prediction,truth,prefix,key,turning=False):
    e=(prediction-truth).square()
    if turning:
        w=turning_weights(prefix,truth,key)
        key_loss=(e[:,0,key]*w).sum()/w.sum()
    else:key_loss=e[...,key].mean()
    return .5*e.mean()+.5*key_loss
