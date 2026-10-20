"""Assessment-facing CSV interface and benchmark dashboard for Team DeepGen."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

from src.config import DATASETS, ROOT, RESULTS_DIR, SAMPLE_DIR
from src.submission import (benchmark_table, evaluate_history, load_selected, options,
                            predict_window, read_input, score_forecasts)


@st.cache_resource
def bundle(ds, horizon, family):
    torch.set_num_threads(2)
    return load_selected(ds, horizon, family)


@st.cache_data(show_spinner=False)
def historical_scores(frame, meta, ds, role, family, maximum, _models):
    return evaluate_history(frame, meta, _models, role, maximum)


def forecast_dates(frame, origin, h, ds):
    if origin+h <= len(frame):
        return frame.index[origin:origin+h]
    if isinstance(frame.index, pd.DatetimeIndex):
        if ds == "finance":
            return pd.bdate_range(frame.index[origin-1]+pd.Timedelta(days=1), periods=h)
        return pd.date_range(frame.index[origin-1]+pd.Timedelta(hours=1), periods=h, freq="h")
    return pd.RangeIndex(origin, origin+h)


def forecast_figure(frame, origin, dates, predictions, scaler, variables, role, lookback):
    fig = make_subplots(rows=len(variables), cols=1, shared_xaxes=True,
                        subplot_titles=variables if len(variables)>1 else None, vertical_spacing=.12 if len(variables)>1 else .02)
    columns = list(frame.columns)
    for row, column in enumerate(variables, 1):
        j = columns.index(column)
        ix = slice(max(0, origin-lookback), origin)
        fig.add_trace(go.Scatter(x=frame.index[ix], y=frame.iloc[ix, j], name="Observed input",
                                line=dict(color="#9aa6b2", width=1.7), showlegend=row==1), row, 1)
        if len(frame) >= origin+len(dates):
            fig.add_trace(go.Scatter(x=dates, y=frame.iloc[origin:origin+len(dates), j], name="Actual future",
                                    mode="lines+markers", line=dict(color="#122a3a", width=2.4), showlegend=row==1), row, 1)
        names = [("source_model", "Source-only", "#d7803e", "dash"),
                 ("adapted_model", "Adapted", "#2478cf", "solid"),
                 ("persistence", "Persistence", "#66a69a", "dot")]
        for key, label, color, dash in names:
            if role == "source" and key == "adapted_model":
                continue
            fig.add_trace(go.Scatter(x=dates, y=scaler.inverse(predictions[key])[0, :, j], name=label,
                                    mode="lines+markers", line=dict(color=color, width=2, dash=dash), showlegend=row==1), row, 1)
    fig.update_layout(template="plotly_white", height=270*len(variables)+65, hovermode="x unified",
                      legend=dict(orientation="h", y=1.1), margin=dict(l=10,r=10,t=70,b=20))
    return fig


def studio(ds, horizon, family, role, meta, models, uploaded):
    samples = sorted(SAMPLE_DIR.glob(f"{ds}_{'domain2' if role=='target' else 'domain1'}_*.csv"))
    file = uploaded if uploaded is not None else samples[0] if samples else None
    if file is None:
        st.info("Upload a CSV to begin.")
        return
    try:
        frame = read_input(file, ds)
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        return
    l, h = meta["L"], meta["H"]
    if len(frame) < l:
        st.error(f"Need at least {l} usable observations; this file has {len(frame)}.")
        return
    st.caption(f"{'Uploaded CSV' if uploaded is not None else 'Bundled sample: '+samples[0].name} · {len(frame):,} usable rows · {len(frame.columns)} forecast variables")
    a, b = st.columns([2, 1])
    after_end = a.checkbox("Forecast after the final uploaded row", value=len(frame)<l+h)
    if after_end:
        origin = len(frame)
    elif len(frame)-h > l:
        origin = a.slider("Forecast origin (first predicted row)", l, len(frame)-h, len(frame)-h,
                          help="Only rows before this origin enter the forecast.")
    else:
        origin = l
    variables = b.multiselect("Variables to display", meta["columns"], default=[meta["key"]])
    variables = variables or [meta["key"]]
    scaler, x, truth, predictions = predict_window(frame, meta, models, role, origin)
    dates = forecast_dates(frame, origin, h, ds)
    task_label = "Next-hour forecast" if h == 1 else "Next 24-hour forecast" if h == 24 else "Next 5-trading-day forecast"
    st.subheader(task_label)
    st.caption(f"{h} future {'hour' if h == 1 else 'hours' if ds != 'finance' else 'trading days'} · {len(meta['columns'])} predicted variables · {family}")
    show = 48 if h == 1 else min(l, 4*h)
    st.plotly_chart(forecast_figure(frame, origin, dates, predictions, scaler, variables, role, show), width="stretch")
    scores = score_forecasts(x, truth, predictions, meta, scaler)
    main = "adapted_model" if role == "target" else "source_model"
    if len(scores):
        st.markdown("**Selected-window performance**")
        cols = st.columns(4)
        for col, key, label in zip(cols, ("key_RMSE","key_MAE","MSE","MAE"),
                                   (f"{meta['key']} RMSE", f"{meta['key']} MAE", "Multivariate MSE", "Multivariate MAE")):
            col.metric(label, f"{scores.loc[main,key]:.4f}", help=meta["key_unit"] if key.startswith('key') else "Standardized across all forecast variables; this window only.")
        st.caption("These scores describe the selected window, not the saved benchmark or the entire uploaded file.")
    else:
        st.info("The forecast extends beyond the file. No actual future values are available to score it.")
    output = pd.DataFrame(scaler.inverse(predictions[main])[0], columns=meta["columns"], index=pd.Index(dates,name="date"))
    with st.expander(f"View all {h} forecast {'timestamp' if h == 1 else 'timestamps'} and variables"):
        st.dataframe(output, width="stretch")
    st.download_button("Download this forecast", output.to_csv().encode(), f"{ds}_{role}_{h}step_forecast.csv", "text/csv")
    st.markdown("---")
    st.subheader("Evaluate historical forecasts")
    st.write("Compare frozen source/adapted models and baselines on future values already present in your CSV.")
    available = max(0,(len(frame)-l-h)//h+1)
    if not available:
        st.caption(f"Provide at least {l+h} usable rows to evaluate a complete forecast.")
        return
    maximum = st.number_input("Maximum forecast origins to evaluate (latest available)", 1, max(1,available), min(1000,available), step=1)
    st.caption("One-hour forecasts refresh observations every hour. Longer forecasts start every H rows and do not receive observations inside their horizon.")
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    request=(ds,horizon,family,role,int(maximum),fingerprint)
    run_evaluation=st.button("Run historical evaluation", type="primary")
    if run_evaluation or st.session_state.get("last_evaluation_request")==request:
        st.session_state["last_evaluation_request"]=request
        with st.spinner("Evaluating frozen forecasts…"):
            summary, export, origins = historical_scores(frame, meta, ds, role, family, int(maximum), models)
        labels = {"source_model":"Source-only", "adapted_model":"Adapted", "persistence":"Persistence", "seasonal_naive":"Seasonal naive", "input_mean":"Input mean"}
        visible = summary.rename(index=labels)[["MSE","MAE","key_RMSE","key_MAE"]]
        st.caption(f"{len(origins):,} origins · first row {origins[0]} · last row {origins[-1]} · H={h}. Every prediction uses only its preceding input.")
        st.dataframe(visible.rename(columns={"key_RMSE":f"{meta['key']} RMSE","key_MAE":f"{meta['key']} MAE"}).style.format("{:.4f}"), width="stretch")
        if role == "target":
            before, after = summary.loc["source_model","key_RMSE"], summary.loc["adapted_model","key_RMSE"]
            if before > 0:
                gain = 100*(1-after/before)
                st.write(f"Adaptation changes {meta['key']} RMSE by {-gain:+.2f}% relative to source-only on this file.")
        chart = export[export.variable==meta["key"]]
        fig = go.Figure()
        for key,label,color in (("actual","Actual","#122a3a"),(main,"Adapted" if role=="target" else "Source-only","#2478cf"),("persistence","Persistence","#66a69a")):
            fig.add_trace(go.Scatter(x=chart.date,y=chart[key],name=label,line=dict(color=color)))
        if role=="target":
            fig.add_trace(go.Scatter(x=chart.date,y=chart.source_model,name="Source-only",line=dict(color="#d7803e",dash="dash")))
        fig.update_layout(template="plotly_white",height=400,title="Rolling next-hour evaluation" if h==1 else "Successive forecasts from separate origins",yaxis_title=meta["key_unit"],hovermode="x unified")
        st.plotly_chart(fig,width="stretch")
        st.download_button("Download evaluated forecasts and actuals",export.to_csv(index=False).encode(),f"{ds}_{role}_evaluation.csv","text/csv")
        st.download_button("Download evaluation metrics",summary.to_csv().encode(),f"{ds}_{role}_metrics.csv","text/csv")


def benchmark(ds, horizon):
    st.subheader("Saved benchmark evidence")
    st.caption("Three-seed means on previously inspected chronological development tests. The original saved LiteTCN reference is seed 0. Uploaded-file scores can differ from these benchmark averages.")
    frame = benchmark_table(ds, horizon)
    target = frame[frame.domain=="target"].copy()
    target["label"] = target.family+" · "+target.method
    fig = px.bar(target,x="key_RMSE",y="label",orientation="h",labels={"key_RMSE":DATASETS[ds]["key"]+" RMSE ("+DATASETS[ds]["key_unit"]+")","label":""},color_discrete_sequence=["#2478cf"])
    fig.update_layout(template="plotly_white",height=max(350,len(target)*34),margin=dict(l=10,r=10,t=10,b=20))
    st.plotly_chart(fig,width="stretch")
    st.dataframe(frame.round(5),width="stretch",hide_index=True)
    if horizon==1:
        peaks = pd.read_csv(RESULTS_DIR/"forecast_horizons"/f"{ds}_next_hour_peak_comparison.csv")
        st.markdown("**Daily event errors from rolling one-hour predictions**")
        st.dataframe(peaks.round(4),width="stretch",hide_index=True)
        st.caption("Each episode groups 24 predictions issued at successive origins, with hourly observation updates. Energy persistence still has the lowest trough error among these highlighted adapted models.")
        st.image(str(RESULTS_DIR/"forecast_horizons"/"figures"/f"{ds}_rolling_next_hour.png"),width="stretch")
    elif ds!="finance":
        st.image(str(RESULTS_DIR/"forecast_horizons"/"figures"/f"{ds}_single_origin_24h.png"),width="stretch")
        st.caption("The TCN/boosting follow-up is a separate comparison; the earlier strict hybrid remains visible above and can be stronger. No hourly updates occur inside a single-origin 24-hour forecast.")
    else:
        st.info("Finance predicts return, high–low range, overnight gap and log volume. The strict model improves overall MSE only about 1.60% over the strong input-mean baseline; no price-level or trading-profit claim is made.")
    st.download_button("Download benchmark table",frame.to_csv(index=False).encode(),f"{ds}_{horizon}step_benchmark.csv","text/csv")


def main():
    st.set_page_config(page_title="DeepGen · Forecast Adaptation",page_icon="🌡️",layout="wide")
    st.markdown("""<style>.block-container{padding-top:4rem;max-width:1350px}h1{letter-spacing:-1px}section[data-testid=stSidebar]{background:#eff4f8}div[data-testid=stMetric]{background:#f1f6fa;padding:14px;border-radius:8px}footer{visibility:hidden}</style>""",unsafe_allow_html=True)
    st.caption("TEAM DEEPGEN · IE643 COURSE PROJECT")
    st.title("Forecasting across domains")
    st.write("Upload a multivariate time series, compare source and adapted forecasts, and inspect the evidence behind the results.")
    with st.sidebar:
        st.header("Forecast setup")
        ds = st.selectbox("Dataset",list(DATASETS),key="dataset",format_func=lambda x:{"climate":"Climate · Shimla → Delhi","energy":"Energy · ETTh1 → ETTh2","finance":"Finance · IT → Pharma"}[x])
    horizons = [1,24] if ds!="finance" else [5]
    horizon = st.radio("Prediction type",horizons,horizontal=True,key=f"horizon_{ds}",
                       format_func=lambda x:"Next 5 trading days" if ds=="finance" else "Next hour" if x==1 else "Next 24 hours",
                       help="This choice changes the trained model, number of forecast timestamps, historical evaluation and saved benchmark results.")
    with st.sidebar:
        family = st.selectbox("Model",options(ds,horizon),key=f"model_{ds}_{horizon}")
        domain = st.radio("Input domain",["Domain 1 · source","Domain 2 · target"],index=1,key="input_domain")
        role = "source" if domain.startswith("Domain 1") else "target"
        meta,models = bundle(ds,horizon,family)
        st.markdown(f"**{meta[role+'_label']}**")
        uploaded = st.file_uploader("Upload CSV",type=["csv"],key=f"upload_{ds}")
        with st.expander("CSV requirements"):
            st.write("Columns:",", ".join(meta["columns"]))
            st.write(f"At least {meta['L']} usable historical rows. Add {meta['H']} actual future rows to evaluate a forecast.")
            st.write("Hourly datasets require consecutive timestamps. Upload one series, in chronological order, with finite observations.")
            if ds=="finance":st.write("Raw Date/Open/High/Low/Close/Volume is also accepted; one extra row is needed to form the first return.")
            if isinstance(uploaded,type(None)):
                sample=next(iter(sorted(SAMPLE_DIR.glob(f"{ds}_{'domain1' if role=='source' else 'domain2'}_*.csv"))),None)
                if sample:st.download_button("Download sample CSV",sample.read_bytes(),sample.name,"text/csv")
        st.caption("Frozen inference · target forecasting labels excluded from adaptation")
    st.info("Each next-hour forecast receives refreshed observations before the next origin. It is a different task from predicting 24 hours at once." if horizon==1 else f"Each forecast predicts {horizon} {'trading days' if ds=='finance' else 'hours'} from one origin, without observation updates inside that horizon.")
    live, evidence, method = st.tabs(["Forecast studio","Benchmark dashboard","Method & downloads"])
    with live:studio(ds,horizon,family,role,meta,models,uploaded)
    with evidence:benchmark(ds,horizon)
    with method:
        st.subheader("Source learning → observed-input adaptation → frozen forecasts")
        st.write("Source training uses historical inputs and source futures. Strict adaptation withholds values inside each individual target input, using its earlier prefix to predict them. Original target forecasting labels and cross-window label reconstruction are excluded.")
        st.write("The peak-calibrated next-hour hybrid retains LiteTCN, local dynamics and earlier forecasting experts. Causal trend regimes select convex weights fitted using observed daily episodes, event-sensitive loss, source-retention guards and fallback.")
        st.warning("Known limitations: energy trough error can worsen with adaptation; persistence remains stronger at those troughs. Day-ahead peaks remain difficult. Tests were reused during development, so these are not independent final-confirmation results.")
        for p,label in ((ROOT/"report/intensive/IE643_Intensive_Assessment_Report.pdf","Technical report (PDF)"),(ROOT/"dashboard/IE643_Results_Dashboard.html","Offline results dashboard"),(ROOT/"report/IE643_ASSESSMENT_GUIDE.html","Complete study guide")):
            if p.exists():st.download_button("Download "+label,p.read_bytes(),p.name,"application/pdf" if p.suffix==".pdf" else "text/html")
        st.markdown("**DeepGen** · Ayush Patel (23B0037) · Rishik Yadav (23B2103)")
        st.caption("Both: research and experiments. Ayush: dashboard/application. Rishik: report/presentation. Team allocation supplied by the team.")


if __name__=="__main__":
    main()
