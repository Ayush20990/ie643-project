from pathlib import Path
import pandas as pd
import src.novelty_finance as n
for h in [1,5]:
 n.H=h;n.OUT=n.ROOT/('results/novelty_finance' if h==1 else 'results/novelty_finance_mean5')
 m=pd.read_csv(n.OUT/'metrics.csv');q=m[m.domain=='target'].groupby('method')[['RMSE','MAE','MSE']].mean()
 old=pd.read_csv(n.OUT/'legacy_matched_metrics.csv').groupby('method')[['RMSE','MAE','MSE']].mean()
 q.loc['Earlier adapted (matched)']=old.loc['Earlier hybrid adapted_model']
 import matplotlib.pyplot as plt
 names=['Persistence','Mean 60','Mean 20','Mean 5','HAR','TCN source','Hybrid source','Hybrid replay','Hybrid both','Earlier adapted (matched)']
 fig,ax=plt.subplots(figsize=(10,4.6));ax.barh(names,[q.loc[k,'RMSE'] for k in names],color=['#9baeb9']*4+['#e3a347']+['#226d94']*3+['#188477','#835b99']);ax.invert_yaxis();ax.set_xlabel(f'{h}-session mean range RMSE (percentage points; lower is better)');ax.set_title('Pharma transfer | three-seed mean | identical forecast origins');fig.tight_layout();fig.savefig(n.OUT/'comparison.png',dpi=180);plt.close(fig)
