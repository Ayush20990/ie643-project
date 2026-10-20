"""Source-only past-weather diagnostic; no actual future weather predictors."""
import numpy as np
import pandas as pd
import torch

from .config import DATASETS, PROC_DIR, RESULTS_DIR
from .data import Scaler, chrono_split, load_domain, make_windows
from .energy_driver_analysis import features, ridge_forecasts, PENALTIES
from .improve import write_json
from .progressive_train import sha


def main():
    torch.set_num_threads(2)
    cfg=DATASETS["climate"]
    # Analysis helper expects the response in the last column; the forecasting
    # pipeline/checkpoints retain their original column order throughout.
    columns=[c for c in cfg["columns"] if c!=cfg["key"]]+[cfg["key"]]
    df=load_domain("climate",cfg["source"]).sort_values("date")
    values=df[columns].to_numpy(dtype=np.float32)
    parts=chrono_split([values],cfg["split"],cfg["L"])
    scaler=Scaler.fit(parts["train"])
    windows={s:tuple(v.numpy() for v in make_windows([scaler.transform(parts[s][0])],cfg["L"],cfg["H"],
                        stride=cfg["train_stride"] if s=="train" else 1)) for s in ("train","val")}
    a,b=int(len(df)*cfg["split"][0]),int(len(df)*sum(cfg["split"][:2]))
    clocks={}
    for split,dates in (("train",df.date.iloc[:a]),("val",df.date.iloc[a-cfg["L"]:b])):
        stride=cfg["train_stride"] if split=="train" else 1
        origin=pd.DatetimeIndex(dates.iloc[cfg["L"]-1:cfg["L"]-1+len(windows[split][0])*stride:stride]).as_unit("ns")
        hours=origin.asi8.astype(np.float64)/3.6e12
        clocks[split]=np.stack([f(2*np.pi*hours/period) for period in (24,168) for f in (np.sin,np.cos)],-1)
    xt,yt=windows["train"];xv,yv=windows["val"]
    residual=yt[...,-1]-xt[:,-1:,-1]
    out=RESULTS_DIR/"coupled_litetcn"/"climate_drivers";out.mkdir(parents=True,exist_ok=True)
    rows,search=[],[]
    for name,weather in (("temperature_history_calendar",False),("temperature_history_calendar_past_weather",True)):
        best=None
        for ridge,delta in ridge_forecasts(features(xt,clocks["train"],weather),residual,features(xv,clocks["val"],weather)):
            error=(delta+xv[:,-1:,-1]-yv[...,-1])*scaler.std[-1]
            row={"model":name,"ridge":ridge,"source_val_24h_RMSE":float(np.sqrt(np.mean(error**2)))}
            search.append(row)
            if best is None or row["source_val_24h_RMSE"]<best[0]["source_val_24h_RMSE"]:best=row,error
        row,error=best
        rows.append({**row,**{f"RMSE_first_{h}h":float(np.sqrt(np.mean(error[:,:h]**2))) for h in (1,3,6,12,24)}})
    train=values[:a];delta=np.diff(train[:,-1]);correlations=[]
    for c,name in enumerate(columns[:-1]):
        for lag in (0,1,3,6,12,24,48):
            correlations.append({"weather_variable":name,"past_lag_hours":lag,
                "correlation_with_next_temperature_change":float(np.corrcoef(train[:len(delta)-lag,c],delta[lag:])[0,1])})
    pd.DataFrame(rows).to_csv(out/"source_validation_forecasts.csv",index=False)
    pd.DataFrame(search).to_csv(out/"ridge_search.csv",index=False)
    pd.DataFrame(correlations).to_csv(out/"training_lag_associations.csv",index=False)
    write_json(out/"protocol.json",{"domain":"Shimla source only","target_values_used":False,"future_weather_inputs_used":False,
        "analysis_column_order":columns,"source_train_rows":a,"source_val_rows":b-a,"penalties":PENALTIES,
        "selection_and_reporting_period":"source validation; no independent confirmation or causal claim",
        "source_file_sha256":sha(PROC_DIR/"climate"/"shimla.csv")})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":main()
