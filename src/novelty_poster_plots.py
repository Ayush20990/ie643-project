"""Poster examples for all three datasets, fixed origins and saved seed-0 predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import DATASETS
ROOT=Path(__file__).resolve().parents[1]

def build():
    out=ROOT/'report/novelty';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(11,8.4))
    specs=[('truth','Actual','#163141','-'),('source_model','Source-only','#dc8534','--'),('adapted_model','Adapted','#16887d','-')]
    evidence=[]
    for ax,ds,title in zip(axes[:2],['climate','energy'],['Weather: Delhi air temperature','Energy: ETTh2 oil temperature']):
        cfg=DATASETS[ds];j=cfg['columns'].index(cfg['key'])
        path=ROOT/f'results/peak_next_hour/{ds}/rolling_forecasts.npz'
        a=np.load(path);meta=json.loads((ROOT/f'models/peak_next_hour/{ds}/meta.json').read_text())
        sd,mu=meta['target_scaler']['std'][j],meta['target_scaler']['mean'][j]
        for name,label,color,style in specs:
            ax.plot(np.arange(96),a[f'target_{name}'][:96,0,j]*sd+mu,label=label,color=color,ls=style,lw=1.8)
        ax.set_title(title+' | rolling next hour',loc='left',fontsize=14,fontweight='bold')
        ax.set_ylabel('Temperature (°C)');ax.set_xlabel('First 96 test origins; observations refreshed every hour',fontsize=11)
        ax.grid(alpha=.18)
        evidence.append(dict(dataset=ds,file=str(path.relative_to(ROOT)),seed=0,start_origin=0,count=96,horizon=1,inverse_scaler='target_scaler'))
    ax=axes[2];path=ROOT/'results/novelty_finance_mean5/predictions.csv';df=pd.read_csv(path)
    q=df[(df.seed==0)&(df.series=='CIPLA.NS')].iloc[:100]
    for name,label,color,style in [('actual','Actual','#163141','-'),('Hybrid source','Source-only','#dc8534','--'),('Hybrid both','Adapted','#16887d','-')]:
        ax.plot(np.arange(len(q)),q[name],label=label,color=color,ls=style,lw=1.6)
    ax.set_title('Finance: CIPLA | next-five-session average range',loc='left',fontsize=14,fontweight='bold')
    ax.set_ylabel('100 ln(High / Low)');ax.set_xlabel('First 100 test origins; overlapping five-session targets',fontsize=11);ax.grid(alpha=.18)
    evidence.append(dict(dataset='finance',file=str(path.relative_to(ROOT)),seed=0,series='CIPLA.NS',count=100,horizon=5,target='mean future high-low log range',first_date=q.date.iloc[0],last_date=q.date.iloc[-1]))
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.53,1.005),frameon=False,fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96],h_pad=1.6)
    fig.savefig(out/'all_three_forecast_examples.png',dpi=200,bbox_inches='tight');plt.close(fig)
    (out/'poster_plot_provenance.json').write_text(json.dumps(evidence,indent=2))
    return out/'all_three_forecast_examples.png'
if __name__=='__main__':build()
