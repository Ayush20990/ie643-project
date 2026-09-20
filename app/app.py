"""Streamlit interface: upload a multivariate CSV from Domain 1 or Domain 2, visualise
input / actual / forecast and see goodness-of-forecast metrics.

Run:  streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATASETS, MODEL_DIR, RESULTS_DIR, SAMPLE_DIR  # noqa: E402
from src.data import Scaler  # noqa: E402
from src.data_download import ohlcv_to_features  # noqa: E402
from src.metrics import all_metrics, seasonal_naive  # noqa: E402
from src.model import LiteTCNForecaster  # noqa: E402

C_INPUT, C_ACTUAL, C_ADAPT, C_SRC, C_NAIVE = "#8a8984", "#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a"

st.set_page_config(page_title="CARE-DA Forecast Adaptation", page_icon="🌡️", layout="wide")


@st.cache_resource
def load_models(ds):
    meta = json.loads((MODEL_DIR / ds / "meta.json").read_text())
    C = len(meta["columns"])
    mods = {}
    names = ["source_model", "adapted_model"]
    if (MODEL_DIR / ds / "adapted_model_v1.pt").exists():
        names.append("adapted_model_v1")
    for name in names:
        m = LiteTCNForecaster(meta["L"], meta["H"], C)
        path = MODEL_DIR / ds / (f"{name}.pt" if name != "adapted_model_v1" else "adapted_model_v1.pt")
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        mods[name] = m
    return meta, mods


def read_csv(file, ds, cols):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    if ds == "finance" and not set(cols).issubset(df.columns):
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            df = ohlcv_to_features(df)
    date_col = next((c for c in df.columns if c.lower() in ("date", "time", "datetime", "timestamp")), None)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    idx = pd.to_datetime(df[date_col]) if date_col else pd.RangeIndex(len(df))
    out = df[cols].astype(float).interpolate(limit_direction="both")
    out.index = idx
    return out


@torch.no_grad()
def forecast(model, xs):
    return model(torch.from_numpy(xs.astype(np.float32))).numpy()


st.title("Unsupervised Domain Adaptation for Multivariate Forecasting")
st.caption("IE 643 · DeepGen · LiteTCN trained on Domain 1, adapted to Domain 2 with CARE-DA v2 "
           "(MMD + forecast-consistent pretext + spectral alignment; no target labels)")

with st.sidebar:
    ready = [k for k in DATASETS if (MODEL_DIR / k / "adapted_model.pt").exists()]
    if not ready:
        st.error("No trained models found. Run `python -m src.run_experiments --dataset <name>` first.")
        st.stop()
    qp = st.query_params.get("ds")
    ds = st.selectbox("Benchmark", ready, index=ready.index(qp) if qp in ready else 0,
                      format_func=lambda k: DATASETS[k]["title"])
    meta, models = load_models(ds)
    dom = st.radio("Input domain", ["Domain 1 (source)", "Domain 2 (target)"],
                   index=1 if st.query_params.get("dom") == "2" else 0)
    is_tgt = dom.startswith("Domain 2")
    st.markdown(f"**{meta['target_label'] if is_tgt else meta['source_label']}**")
    role = "domain2" if is_tgt else "domain1"
    samples = sorted(SAMPLE_DIR.glob(f"{ds}_{role}_*.csv"))
    up = st.file_uploader("Upload multivariate CSV", type="csv")
    src_file = up if up is not None else (samples[0] if samples else None)
    if up is None and samples:
        st.info(f"Using bundled sample: {samples[0].name}")
    with st.expander("Expected CSV format"):
        st.write("A `date` column plus these columns:", meta["columns"])
        if ds == "finance":
            st.write("…or raw Yahoo-style `Date, Open, High, Low, Close, Volume` (features are derived).")
        st.write(f"At least L = {meta['L']} rows (look-back); L + H = {meta['L'] + meta['H']} rows to "
                 "compare against actual values.")

tab_live, tab_bench = st.tabs(["Forecast your data", "Benchmark results"])

with tab_live:
    if src_file is None:
        st.warning("Upload a CSV to begin.")
        st.stop()
    cols, L, H = meta["columns"], meta["L"], meta["H"]
    try:
        df = read_csv(src_file, ds, cols)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()
    if len(df) < L:
        st.error(f"Need at least {L} rows, got {len(df)}.")
        st.stop()

    sc = Scaler(**meta["target_scaler" if is_tgt else "source_scaler"])
    raw = df.values.astype(np.float32)
    z = sc.transform(raw)
    main_model = models["adapted_model" if is_tgt else "source_model"]
    key_idx = cols.index(meta["key"])

    max_start = len(df) - L - (H if len(df) >= L + H else 0)
    c1, c2 = st.columns([3, 2])
    start = c1.slider("Start of look-back window (row index)", 0, max(0, max_start), max(0, max_start))
    show = c2.multiselect("Variables to plot", cols, default=cols[: min(4, len(cols))])
    c3, c4 = st.columns([3, 2])
    n_show = c3.slider("Look-back steps to display", min(L, H), L, min(L, 4 * H),
                       help="The model always uses the full look-back window; this only changes the plot.")
    compare = is_tgt and c4.checkbox("Overlay un-adapted source and CARE-DA v1", value=True)

    x = z[start:start + L][None]
    has_actual = start + L + H <= len(df)
    p_main = sc.inverse(forecast(main_model, x)[0])
    p_src = sc.inverse(forecast(models["source_model"], x)[0]) if compare else None
    p_v1 = None
    if compare and "adapted_model_v1" in models:
        p_v1 = sc.inverse(forecast(models["adapted_model_v1"], x)[0])
    t_in = df.index[start + L - n_show:start + L]
    if has_actual:
        t_out = df.index[start + L:start + L + H]
    else:
        step = (df.index[-1] - df.index[-2]) if len(df) > 1 else 1
        t_out = [df.index[-1] + step * (i + 1) for i in range(H)]

    fig = make_subplots(rows=len(show) or 1, cols=1, shared_xaxes=True, subplot_titles=show,
                        vertical_spacing=0.06)
    for r, c in enumerate(show, start=1):
        j = cols.index(c)
        first = r == 1
        fig.add_trace(go.Scatter(x=t_in, y=raw[start + L - n_show:start + L, j], name="Input (look-back)",
                                 line=dict(color=C_INPUT, width=2), showlegend=first, legendgroup="in"), r, 1)
        if has_actual:
            fig.add_trace(go.Scatter(x=t_out, y=raw[start + L:start + L + H, j], name="Actual",
                                     line=dict(color=C_ACTUAL, width=2), showlegend=first, legendgroup="act"), r, 1)
        if p_src is not None:
            fig.add_trace(go.Scatter(x=t_out, y=p_src[:, j], name="Source-only forecast",
                                     line=dict(color=C_SRC, width=2, dash="dash"), showlegend=first,
                                     legendgroup="src"), r, 1)
        if p_v1 is not None:
            fig.add_trace(go.Scatter(x=t_out, y=p_v1[:, j], name="CARE-DA v1 forecast",
                                     line=dict(color="#4a3aa7", width=2, dash="dot"), showlegend=first,
                                     legendgroup="v1"), r, 1)
        fig.add_trace(go.Scatter(x=t_out, y=p_main[:, j],
                                 name="CARE-DA v2 adapted forecast" if is_tgt else "Source model forecast",
                                 line=dict(color=C_ADAPT, width=2), showlegend=first, legendgroup="main"), r, 1)
        fig.add_vline(x=t_out[0], line=dict(color="#c3c2b7", width=1, dash="dot"), row=r, col=1)
    fig.update_layout(height=230 * max(1, len(show)) + 60, hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", yref="container", y=0.995, yanchor="top", x=0),
                      margin=dict(l=10, r=10, t=75, b=10))
    st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------ metrics
    st.subheader("Goodness of forecast")
    if has_actual:
        y = z[start + L:start + L + H][None]
        m_win = all_metrics(x, y, forecast(main_model, x), meta["season"], sc, key_idx)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Window MSE (z-scored)", f"{m_win['MSE']:.3f}")
        k2.metric("Window MAE (z-scored)", f"{m_win['MAE']:.3f}")
        k3.metric("Window MASE", f"{m_win['MASE']:.3f}", help="<1 means better than the seasonal-naive forecast")
        k4.metric(f"{meta['key']} RMSE", f"{m_win['key_RMSE']:.3f}", help=meta["key_unit"])
    else:
        st.info("Forecast extends beyond the file - no actual values to score this window.")

    n_win = (len(df) - L - H) // H + 1 if len(df) >= L + H else 0
    if n_win >= 1:
        idx = [i * H for i in range(n_win)]
        X = np.stack([z[i:i + L] for i in idx])
        Y = np.stack([z[i + L:i + L + H] for i in idx])
        rows = {"CARE-DA v2 adapted" if is_tgt else "Source model": forecast(main_model, X)}
        if is_tgt:
            rows["Source-only (no adaptation)"] = forecast(models["source_model"], X)
            if "adapted_model_v1" in models:
                rows["CARE-DA v1"] = forecast(models["adapted_model_v1"], X)
        rows["Seasonal naive"] = seasonal_naive(X, H, meta["season"])
        table = pd.DataFrame({k: all_metrics(X, Y, v, meta["season"], sc, key_idx) for k, v in rows.items()}).T
        table = table.rename(columns={"key_RMSE": f"{meta['key']} RMSE", "key_MAE": f"{meta['key']} MAE"})
        st.markdown(f"**Whole-file rolling evaluation** — {n_win} non-overlapping windows (L={L}, H={H})")
        st.dataframe(table.style.format("{:.4f}").highlight_min(axis=0, color="#d6e6f9"), width="stretch")
        if is_tgt:
            a, s = table.iloc[0]["MSE"], table.iloc[1]["MSE"]
            st.success(f"Adaptation changes MSE by {100 * (a - s) / s:+.1f}% relative to the un-adapted model.")

    out = pd.DataFrame(p_main, columns=cols, index=pd.Index(t_out, name="date"))
    st.download_button("Download forecast CSV", out.to_csv().encode(), f"forecast_{ds}_{role}.csv", "text/csv")

with tab_bench:
    v2c = RESULTS_DIR / ds / "care_v2_compare.csv"
    if v2c.exists():
        st.markdown("**CARE-DA v2 vs v1 vs source-only** (seed-0 target test, no target labels used).")
        st.dataframe(pd.read_csv(v2c, index_col=0).style.format("{:.4f}"), width="stretch")
    summ = RESULTS_DIR / ds / "summary_target.csv"
    if summ.exists():
        s = pd.read_csv(summ, index_col=0)
        show_cols = [c for c in s.columns if c.endswith("_mean")]
        st.markdown("Full v1 benchmark — target-domain held-out test metrics, mean over 3 seeds. "
                    "`oracle_finetune` and `target_only` use target labels (reference only).")
        st.dataframe(s[show_cols].rename(columns=lambda c: c[:-5]).style.format("{:.4f}"),
                     width="stretch")
    figs = sorted((RESULTS_DIR / "figures").glob(f"{ds}_*.png"))
    extra = RESULTS_DIR / "figures" / "care_v2_compare.png"
    if extra.exists():
        figs.append(extra)
    for f in figs:
        st.image(str(f), caption=f.stem, width="stretch")
