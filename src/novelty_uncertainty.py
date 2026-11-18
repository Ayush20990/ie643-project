"""Paired moving-block bootstrap; resample dates jointly across companies and seeds."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 for suffix in ['novelty_finance','novelty_finance_mean5']:
  folder=ROOT/'results'/suffix;d=pd.read_csv(folder/'predictions.csv')
  rows=[];rng=np.random.default_rng(20261118)
  for base in ['Persistence','Mean 60','Hybrid source','Hybrid replay']:
   q=d.assign(delta=(d.actual-d[base])**2-(d.actual-d['Hybrid both'])**2).groupby('date').delta.mean().to_numpy()
   n=len(q);samples=[]
   for _ in range(2000):
    starts=rng.integers(0,n-20+1,size=int(np.ceil(n/20)));ids=np.concatenate([np.arange(s,s+20) for s in starts])[:n]
    samples.append(q[ids].mean())
   lo,hi=np.quantile(samples,[.025,.975])
   rows.append(dict(baseline=base,mean_MSE_reduction=float(q.mean()),CI95_lower=float(lo),CI95_upper=float(hi),block_sessions=20,bootstrap_replicates=2000,dates=n))
  pd.DataFrame(rows).to_csv(folder/'block_uncertainty.csv',index=False)
  print(suffix,pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
