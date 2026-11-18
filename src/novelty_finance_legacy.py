"""Compare the recovered H5 model's FIRST step against the new H1 task on identical origins."""
import json
import numpy as np
import pandas as pd
import torch
from src.novelty_finance import ROOT,OUT,windows,metrics
from src.data import Scaler
from src.strict_models import StrictEnsemble
from src.inference import forecast_batches

def main(horizon=1):
 import src.novelty_finance as nf
 nf.H=horizon
 out=ROOT/("results/novelty_finance" if horizon==1 else "results/novelty_finance_mean5")
 torch.set_num_threads(4)
 folder=ROOT/'models/strict_refined/finance'
 meta=json.loads((folder/'meta.json').read_text())
 x,y,dates,names=windows('pharma')['test']
 scaler=Scaler(**meta['target_scaler']); xx=scaler.transform(x)
 rows=[];frames=[]
 for seed in [0,1,2]:
  for name in ['source_model','adapted_model']:
   model=StrictEnsemble(meta,meta['model_spec']['neural_spec'])
   model.load_state_dict(torch.load(folder/f'seed_{seed}'/f'{name}.pt',weights_only=True))
   p=scaler.inverse(forecast_batches(model.eval(),torch.from_numpy(xx)))[:,:horizon,1].mean(1)
   rows.append(dict(seed=seed,method='Earlier hybrid '+name,**metrics(y,p)))
   frames.append(pd.DataFrame(dict(seed=seed,method=name,series=names,date=dates,actual=y,prediction=p)))
 pd.DataFrame(rows).to_csv(out/'legacy_matched_metrics.csv',index=False)
 pd.concat(frames).to_csv(out/'legacy_matched_predictions.csv',index=False)
 print(pd.DataFrame(rows).groupby('method')[['RMSE','MAE','MSE']].mean())
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=int,default=1,choices=[1,5]);main(parser.parse_args().horizon)
