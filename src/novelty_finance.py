"""Next-session range: source-trained HAR + compact TCN, label-free feature adaptation.
Run: python -m src.novelty_finance
Selection uses source validation only. Target Y is used only in final evaluation.
"""
from pathlib import Path
import copy, hashlib, json, time
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/novelty_finance'; DEST=ROOT/'models/novelty_finance'
COLS=['log_return','hl_range','gap','log_volume']; L=60; H=1; SOURCE_EPOCHS=24; ADAPT_EPOCHS=10


def windows(domain):
    df=pd.read_csv(ROOT/f'data/processed/finance/{domain}.csv')
    out={k:[] for k in ['train','val','test']}
    for name,g in df.groupby('series',sort=True):
        g=g.sort_values('date'); x=g[COLS].to_numpy(dtype=np.float32)
        a,b=int(.7*len(g)),int(.8*len(g))
        for key,start,end in [('train',L,a),('val',a,b),('test',b,len(g))]:
            ids=np.arange(start,end-H+1)
            out[key].append((np.stack([x[i-L:i] for i in ids]),np.array([x[i:i+H,1].mean() for i in ids],dtype=np.float32),g.date.iloc[ids].to_numpy(),np.repeat(name,len(ids))))
    return {k:tuple(np.concatenate([v[j] for v in vs]) for j in range(4)) for k,vs in out.items()}


def prepare_x(x):
    # All scaling is computed from preceding observations, independently for each origin.
    anchor=np.maximum(x[:,:,1].mean(1),.05)
    z=np.zeros_like(x)
    z[:,:,0]=x[:,:,0]/anchor[:,None]
    z[:,:,1]=x[:,:,1]/anchor[:,None]
    z[:,:,2]=x[:,:,2]/anchor[:,None]
    v=x[:,:,3]; z[:,:,3]=(v-v.mean(1,keepdims=True))/np.maximum(v.std(1,keepdims=True),.1)
    return np.clip(z,-15,15).astype('float32'),anchor.astype('float32')


def har_features(x):
    r=x[:,:,1]
    feats=[np.ones(len(x))]
    for k in (1,5,20,60): feats.append(r[:,-k:].mean(1))
    for k in (5,20,60): feats.append(r[:,-k:].std(1))
    for k in (1,5,20):
        feats.extend([np.abs(x[:,-k:,0]).mean(1),np.abs(x[:,-k:,2]).mean(1)])
    return np.stack(feats,1).astype('float32')


class RangeTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Conv1d(4,16,1)
        self.layers=nn.ModuleList([nn.Conv1d(16,16,3,dilation=d) for d in (1,2,4)])
        self.head=nn.Linear(16,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,x,features=False):
        h=torch.nn.functional.gelu(self.proj(x.transpose(1,2)))
        for d,conv in zip((1,2,4),self.layers):
            h=h+torch.nn.functional.gelu(conv(torch.nn.functional.pad(h,(2*d,0))))
        z=h[:,:,-1]
        # Positive range ratio, initialized at recent 5-session mean.
        p=(x[:,-5:,1].mean(1)+self.head(z).squeeze(-1)).clamp_min(.02)
        return (p,z) if features else p


def alignment(s,t,kind):
    if kind=='coral':
        s=s-s.mean(0); t=t-t.mean(0)
        return ((s.T@s/(len(s)-1)-t.T@t/(len(t)-1))**2).sum()/(4*s.shape[1]**2)
    z=torch.cat([s,t]); dist=torch.cdist(z,z).square()
    bw=dist.detach().median().clamp_min(1e-4)
    kernel=sum(torch.exp(-dist/(2*bw*a)) for a in (.25,.5,1,2,4))/5
    n=len(s)
    return kernel[:n,:n].mean()+kernel[n:,n:].mean()-2*kernel[:n,n:].mean()


def predict_net(m,z):
    m.eval()
    with torch.no_grad():return np.concatenate([m(torch.from_numpy(q)).numpy() for q in np.array_split(z,max(1,int(np.ceil(len(z)/512))))])


def metrics(y,p):
    return dict(MSE=float(np.mean((y-p)**2)),RMSE=float(np.sqrt(np.mean((y-p)**2))),MAE=float(np.mean(abs(y-p))))


def train(seed,train_x,train_y,val_x,val_y,target_x,kind='source',initial=None):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    tr,a=prepare_x(train_x); va,av=prepare_x(val_x)
    tx,_=prepare_x(target_x)
    m=RangeTCN() if initial is None else copy.deepcopy(initial)
    epochs=SOURCE_EPOCHS if initial is None else ADAPT_EPOCHS
    opt=torch.optim.AdamW(m.parameters(),lr=.001 if initial is None else .0002,weight_decay=.0001)
    initial_state=copy.deepcopy(m.state_dict()); best=copy.deepcopy(initial_state)
    best_score=np.mean((predict_net(m,va)*av-val_y)**2)
    logs=[]
    for epoch in range(epochs):
        m.train(); running=[]
        for ids in np.array_split(rng.permutation(len(tr)),int(np.ceil(len(tr)/256))):
            xs=torch.from_numpy(tr[ids]); yt=torch.from_numpy(train_y[ids]); scale=torch.from_numpy(a[ids])
            p,z=m(xs,True); loss=((p*scale-yt)**2).mean()
            if initial is not None:
                ti=rng.integers(len(tx),size=len(ids)); _,tz=m(torch.from_numpy(tx[ti]),True)
                reg=0
                if kind in ('coral','both'):reg=reg+.1*alignment(z,tz,'coral')
                if kind in ('mmd','both'):reg=reg+.1*alignment(z,tz,'mmd')
                anchor=sum((p-initial_state[n]).square().mean() for n,p in m.named_parameters())
                loss=loss+reg+.01*anchor
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1); opt.step(); running.append(loss.item())
        score=float(np.mean((predict_net(m,va)*av-val_y)**2))
        if score<best_score:best_score=score; best=copy.deepcopy(m.state_dict())
        logs.append(dict(epoch=epoch+1,training_loss=float(np.mean(running)),source_validation_MSE=score))
    m.load_state_dict(best); m.eval()
    return m,logs


def fit_har(x,y,xv,yv):
    z,a=prepare_x(x); zv,av=prepare_x(xv)
    f=har_features(z)*a[:,None]; fv=har_features(zv)*av[:,None]
    best=None
    for alpha in (.001,.01,.1,1,10,100):
        w=np.linalg.solve(f.T@f+alpha*np.eye(f.shape[1]),f.T@y)
        score=np.mean((np.maximum(fv@w,.01)-yv)**2)
        if best is None or score<best[0]:best=(score,alpha,w)
    return best[2],best[1]


def infer(x,bundle,folder=None):
    folder=DEST if folder is None else folder
    z,a=prepare_x(x); har=np.maximum(har_features(z)@np.array(bundle['har_weights'])*a,.01)
    result={'Persistence':x[:,-1,1], 'Mean 60':x[:,:,1].mean(1),
            'Mean 20':x[:,-20:,1].mean(1),'Mean 5':x[:,-5:,1].mean(1),'HAR':har}
    for kind in ['source','replay','coral','mmd','both']:
        model=RangeTCN(); model.load_state_dict(torch.load(folder/f"seed_{bundle['seed']}_{kind}.pt",weights_only=True))
        result['TCN '+kind]=predict_net(model,z)*a
        beta=bundle['blend'] # Same source-selected blend used for every adaptation ablation.
        result['Hybrid '+kind]=(1-beta)*har+beta*result['TCN '+kind]
    return result


def main():
    started=time.time(); torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True); DEST.mkdir(exist_ok=True,parents=True)
    source=windows('it'); target=windows('pharma')
    # Fixed protocol is written before training / evaluation.
    protocol=dict(task=f'Mean high-low log range across next {H} session(s): mean(100*ln(High/Low))',horizon=H,lookback=60,
        split=[.7,.1,.2],seeds=[0,1,2],source_epochs=SOURCE_EPOCHS,adaptation_epochs=ADAPT_EPOCHS,
        adaptation='Source replay MSE + 0.1 MMD + 0.1 CORAL + 0.01 source-weight anchor',
        target_labels_used_for_training=False,target_labels_used_for_selection=False,
        selection='Source validation only; same source-selected blend for all adaptation ablations',
        target_inputs='Target training window inputs only. No target validation or test inputs used for fitting.',
        normalization='Per-origin scale from 60 preceding range observations; identical source/target preprocessing',
        test_status='Previously inspected 2022-2025 development period, not independent confirmation',
        comparison='H1 range-only metrics cannot establish improvement over H5 multivariate MSE',
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/processed/finance').glob('*.csv')})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    allrows=[]; allpred=[]
    w,alpha=fit_har(source['train'][0],source['train'][1],source['val'][0],source['val'][1])
    for seed in protocol['seeds']:
        print('Seed',seed,flush=True)
        m,logs=train(seed,*source['train'][:2],*source['val'][:2],target['train'][0])
        models={'source':m}; training={'source':logs}
        for kind in ['replay','coral','mmd','both']:
            models[kind],training[kind]=train(seed,*source['train'][:2],*source['val'][:2],target['train'][0],kind,m)
            print(' ',kind,'complete',flush=True)
        z,a=prepare_x(source['val'][0]); h=np.maximum(har_features(z)@w*a,.01); p=predict_net(m,z)*a
        scores=[np.mean(((1-b)*h+b*p-source['val'][1])**2) for b in (0,.25,.5,.75,1)]
        beta=[0,.25,.5,.75,1][int(np.argmin(scores))]
        bundle=dict(seed=seed,har_weights=w.tolist(),ridge_alpha=alpha,blend=beta,columns=COLS,L=L,H=H)
        (DEST/f'seed_{seed}.json').write_text(json.dumps(bundle,indent=2))
        for kind,model in models.items():torch.save(model.state_dict(),DEST/f'seed_{seed}_{kind}.pt')
        (OUT/f'training_seed_{seed}.json').write_text(json.dumps(training,indent=2))
        # Evaluation happens after every selection and fit for this seed is complete.
        for domain,parts in [('source',source),('target',target)]:
            x,y,dates,names=parts['test']; predictions=infer(x,bundle)
            for method,pred in predictions.items():
                allrows.append(dict(domain=domain,seed=seed,method=method,**metrics(y,pred)))
                if domain=='target':
                    for company in np.unique(names):
                        mask=names==company
                        allrows.append(dict(domain=company,seed=seed,method=method,**metrics(y[mask],pred[mask])))
            if domain=='target':
                frame=pd.DataFrame(dict(date=dates,series=names,actual=y,seed=seed,**predictions));allpred.append(frame)
    scores=pd.DataFrame(allrows); scores.to_csv(OUT/'metrics.csv',index=False)
    pd.concat(allpred).to_csv(OUT/'predictions.csv',index=False)
    summary=scores.groupby(['domain','method'])[['RMSE','MAE','MSE']].agg(['mean','std'])
    summary.to_csv(OUT/'summary.csv')
    target_scores=scores[scores.domain=='target'].groupby('method')[['RMSE','MAE','MSE']].mean()
    print(target_scores.to_string(),flush=True)
    (OUT/'runtime.json').write_text(json.dumps(dict(seconds=time.time()-started,neural_parameters=sum(p.numel() for p in RangeTCN().parameters())),indent=2))
    plot(pd.concat(allpred),target_scores)


def plot(pred,scores):
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    names=['Persistence','Mean 60','Mean 20','Mean 5','HAR','TCN source','TCN both','Hybrid source','Hybrid both']
    fig,ax=plt.subplots(figsize=(10,4.6)); ax.barh(names,[scores.loc[n,'RMSE'] for n in names],color=['#9baeb9']*4+['#e3a347']+['#226d94','#188477','#226d94','#188477'])
    ax.invert_yaxis();ax.set_xlabel(f'{H}-session mean range RMSE (percentage points; lower is better)');ax.set_title('Pharma transfer | mean of three seed-wise test RMSEs');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    q=pred[(pred.seed==0)&(pred.series=='CIPLA.NS')].iloc[:100]
    fig,ax=plt.subplots(figsize=(11,4));
    for n,c in [('actual','#162b36'),('Mean 20','#b4a08a'),('Hybrid source','#dc8534'),('Hybrid both','#16887d')]:ax.plot(pd.to_datetime(q.date),q[n],label=n,color=c,lw=1.5)
    ax.set_ylabel('100 ln(High / Low)');ax.set_title('Fixed illustration: first 100 CIPLA test origins, seed 0');ax.legend(ncol=4);fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/'forecast.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=int,choices=[1,5],default=1)
    args=parser.parse_args();H=args.horizon
    if H==5:
        OUT=ROOT/'results/novelty_finance_mean5';DEST=ROOT/'models/novelty_finance_mean5'
        SOURCE_EPOCHS=16;ADAPT_EPOCHS=6
    main()
