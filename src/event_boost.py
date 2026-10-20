"""Window-local boosted trajectory/geometry forecasting; no future driver inputs."""
import numpy as np
import torch
from torch import nn
from scipy.fft import dct, idct

from .peak_model import PeakLiteTCN
from .residual_tcn import ResidualLiteTCN

VARIANTS=("trajectory","geometry")
GEOMETRY_COEFFICIENTS=16


def input_pair(packed,horizon,channels):
    if packed.shape[1] <= horizon:
        raise ValueError("Need a real prefix and observed tail")
    return packed[:,:-horizon],packed[:,-horizon:,:channels].copy()


def features(packed,cfg):
    a=np.asarray(packed,dtype=np.float32)
    c,h=len(cfg["columns"]),cfg["H"]
    k=cfg["columns"].index(cfg["key"])
    context=cfg["L"]-h
    if a.shape[1]<context or a.shape[-1]!=c+6:
        raise ValueError("Need real context and timestamp features")
    x=a[:,-context:,:c]
    mean=x.mean(1,keepdims=True)
    sd=x.std(1,keepdims=True).clip(.05)
    z=(x-mean)/sd
    # Preserve key trajectory, multivariate recent state, slopes, cycles and levels.
    parts=[z[...,k],mean[:,0],np.log(sd[:,0]),a[:,-1,c:]]
    for lag in (0,1,2,3,6,12,23,24,47,48,71,95,119,143):
        if lag<context:parts.append(z[:,-lag-1])
    for w in (3,6,12,24,48,72,context):
        parts.extend((z[:,-w:].mean(1),z[:,-w:].std(1)))
    cycles=z[...,k].reshape(len(z),-1,24)
    parts.extend((cycles.mean(2),cycles.max(2),cycles.min(2),cycles.argmax(2)/24.,cycles.argmin(2)/24.))
    return np.concatenate(parts,1).astype(np.float32),mean[:,0,k],sd[:,0,k]


def encode(y,offset,scale,variant):
    z=(np.asarray(y)-offset[:,None])/scale[:,None]
    if variant=="trajectory":return z.astype(np.float32)
    if variant!="geometry":raise ValueError(variant)
    h=z.shape[1]
    high,low=z.argmax(1),z.argmin(1)
    phase=np.stack((np.sin(2*np.pi*high/h),np.cos(2*np.pi*high/h),
                    np.sin(2*np.pi*low/h),np.cos(2*np.pi*low/h)),1)
    # Geometry labels explicitly distinguish E[max Y] from max E[Y].
    return np.concatenate((dct(z,norm="ortho",axis=1)[:,:GEOMETRY_COEFFICIENTS]/np.sqrt(h),
                           2*z.max(1,keepdims=True),2*z.min(1,keepdims=True),phase),1).astype(np.float32)


def decode(raw,offset,scale,variant,mode="mean",horizon=24):
    raw=np.asarray(raw)
    if variant=="trajectory":z=raw[:,:horizon].copy()
    else:
        coeff=np.zeros((len(raw),horizon))
        coeff[:,:GEOMETRY_COEFFICIENTS]=raw[:,:GEOMETRY_COEFFICIENTS]*np.sqrt(horizon)
        z=idct(coeff,norm="ortho",axis=1)
        if mode in ("extrema","phase_extrema"):
            lo,hi=z.min(1),z.max(1)
            desired_hi=raw[:,GEOMETRY_COEFFICIENTS]/2
            desired_lo=raw[:,GEOMETRY_COEFFICIENTS+1]/2
            desired_lo,desired_hi=np.minimum(desired_lo,desired_hi),np.maximum(desired_lo,desired_hi)
            mid=(lo+hi)/2
            wanted_mid=.5*mid+.25*(desired_lo+desired_hi)
            gain=(.5+.5*(desired_hi-desired_lo)/(hi-lo).clip(1e-4)).clip(.5,2.)
            z=(z-mid[:,None])*gain[:,None]+wanted_mid[:,None]
        if mode=="phase_extrema":
            phase=raw[:,GEOMETRY_COEFFICIENTS+2:GEOMETRY_COEFFICIENTS+6]
            peak=np.arctan2(phase[:,0],phase[:,1])*horizon/(2*np.pi)%horizon
            trough=np.arctan2(phase[:,2],phase[:,3])*horizon/(2*np.pi)%horizon
            grid=np.arange(horizon)
            for i in range(len(z)):
                old=[];new=[]
                for actual,wanted in ((int(z[i].argmax()),peak[i]),(int(z[i].argmin()),trough[i])):
                    if 1<=actual<horizon-1:
                        shift=(wanted-actual+horizon/2)%horizon-horizon/2
                        old.append(actual);new.append(np.clip(actual+.5*np.clip(shift,-3,3),.1,horizon-1.1))
                order=np.argsort(old)
                old=np.r_[0,np.asarray(old)[order],horizon-1]
                new=np.r_[0,np.asarray(new)[order],horizon-1]
                if (np.diff(new)>.05).all():
                    z[i]=np.interp(np.interp(grid,new,old),grid,z[i])
    return (z*scale[:,None]+offset[:,None]).astype(np.float32)


def event_quality(y,p):
    y,p=np.asarray(y),np.asarray(p)
    i=np.arange(len(y));hi=y.argmax(1);lo=y.argmin(1)
    mse=float(np.mean((p-y)**2))
    events=float(.5*np.mean((p[i,hi]-y[i,hi])**2+(p[i,lo]-y[i,lo])**2))
    extrema=float(.5*np.mean((p.max(1)-y.max(1))**2+(p.min(1)-y.min(1))**2))
    ramp=float(np.mean((np.diff(p,axis=1)-np.diff(y,axis=1))**2))
    return {"MSE":mse,"event_MSE":events,"extrema_MSE":extrema,"ramp_MSE":ramp,
            "score":mse+.5*events+.25*extrema+.1*ramp}


class EventBoostForecaster(nn.Module):
    """Keep the multivariate neural backbone; a boosted key forecast may replace it."""
    def __init__(self,cfg,base_kind,base_spec,tree,selection):
        super().__init__()
        self.cfg=dict(cfg)
        self.L,self.H,self.C=cfg["L"],cfg["H"],len(cfg["columns"])
        self.key=cfg["columns"].index(cfg["key"])
        self.requires_calendar=True
        self.base_kind=base_kind
        self.base=PeakLiteTCN(cfg,**base_spec) if base_kind=="peak" else ResidualLiteTCN(cfg)
        self.tree,self.selection=tree,dict(selection)

    @torch.no_grad()
    def base_prediction(self,x):
        self.base.eval()
        if self.base_kind=="peak":return self.base(x)
        values=x[...,:self.C]
        if values.shape[1]<self.L:
            pad=values[:,:1].expand(-1,self.L-values.shape[1],-1)
            values=torch.cat((pad,values),1)
        return self.base(values)

    @torch.no_grad()
    def forward(self,packed):
        p=self.base_prediction(packed).clone()
        weight=self.selection["weight"]
        if weight:
            f,offset,scale=features(packed.detach().cpu().numpy(),self.cfg)
            raw=self.tree.predict(f,ntree_end=self.selection["trees"],thread_count=2)
            t=decode(raw,offset,scale,self.selection["variant"],self.selection["mode"],self.H)
            p[...,self.key]=(1-weight)*p[...,self.key]+weight*torch.from_numpy(t).to(p)
        return p
