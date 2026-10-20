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
from src.inference import load_bundle, load_dlinear_bundle, load_strict_bundle, load_peak_bundle, load_residual_bundle, load_progressive_bundle, load_coupled_bundle, load_next_hour_bundle, load_next_hour_reference_bundle, load_peak_next_hour_bundle, load_event_boost_bundle, read_forecast_csv, forecast_batches  # noqa: E402
from src.metrics import all_metrics, seasonal_naive  # noqa: E402

C_INPUT, C_ACTUAL, C_ADAPT, C_SRC, C_NAIVE = "#8a8984", "#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a"

st.set_page_config(page_title="IE643 Forecast Adaptation", page_icon="🌡️", layout="wide")


@st.cache_resource
def load_models(ds, improved=False, transfer=False, dlinear=False, strict=False, context=False, peak=False, local=False, residual=False, progressive=False, coupled=False, next_hour=False, next_hybrid=False, event=False, next_reference=False, peak_next=False):
    meta, models = (load_peak_next_hour_bundle(ds) if peak_next else load_next_hour_reference_bundle(ds) if next_reference else load_event_boost_bundle(ds) if event else load_next_hour_bundle(ds,next_hybrid) if next_hour else load_coupled_bundle(ds, (RESULTS_DIR / "coupled_transfer" / ds / "metrics_all.csv").exists()) if coupled else load_progressive_bundle(ds) if progressive else load_residual_bundle(ds) if residual else load_peak_bundle(ds, local) if peak else load_strict_bundle(ds, context) if strict else
                    load_dlinear_bundle(ds) if dlinear else load_bundle(ds, improved, transfer))
    if improved:
        _, legacy = load_bundle(ds, False)
        models["legacy_source"] = legacy["source_model"]
        models["legacy_adapted"] = legacy["adapted_model"]
    return meta, models


def read_csv(file, ds, cols):
    return read_forecast_csv(file, ds, cols)


@torch.no_grad()
def forecast(model, xs, dates=None):
    return forecast_batches(model, xs, dates=dates)


st.title("Unsupervised Domain Adaptation for Multivariate Forecasting")
st.caption("IE 643 · Multivariate forecasting and domain adaptation · Compare forecasts with held-out actual values and simple baselines")

with st.sidebar:
    ready = [k for k in DATASETS if (MODEL_DIR / k / "adapted_model.pt").exists()]
    if not ready:
        st.error("No trained models found. Run `python -m src.run_experiments --dataset <name>` first.")
        st.stop()
    qp = st.query_params.get("ds")
    ds = st.selectbox("Benchmark", ready, index=ready.index(qp) if qp in ready else 0,
                      format_func=lambda k: DATASETS[k]["title"])
    improved_available = (MODEL_DIR / "improved" / ds / "meta.json").exists()
    families = ["Validation-selected experiments", "Original CARE-DA v2"] if improved_available else ["Original CARE-DA v2"]
    if (MODEL_DIR / "transfer_selection" / ds / "meta.json").exists():
        families.insert(0, "Selected using target history")
        families.insert(1, "DLinear transfer experiment")
    if (RESULTS_DIR / "strict_refined" / ds / "metrics_all.csv").exists():
        families.insert(0, "Strict label-free ensemble")
    if (RESULTS_DIR / "strict_context" / ds / "metrics_all.csv").exists():
        families.insert(1, "Strict shorter-context ensemble")
    if (RESULTS_DIR / "peak_litetcn" / ds / "metrics_all.csv").exists():
        families.insert(0, "LiteTCN v3")
    if (RESULTS_DIR / "peak_calibrated" / ds / "metrics_all.csv").exists():
        families.insert(1, "LiteTCN v3 + recent residuals")
    if (RESULTS_DIR / "residual_litetcn" / ds / "metrics_all.csv").exists():
        families.insert(0 if ds == "energy" else 1, "LiteTCN-R (original encoder)")
    if (RESULTS_DIR / "progressive_litetcn" / ds / "metrics_all.csv").exists():
        families.insert(2, "LiteTCN-R gradual unfreezing")
    if (RESULTS_DIR / "coupled_litetcn" / ds / "metrics_all.csv").exists():
        families.insert(3, "LiteTCN predicted-driver decoder")
    if (RESULTS_DIR / "event_boost" / ds / "metrics_all.csv").exists():
        families.insert(0, "24-hour peak-aware boosted adaptation")
    if (RESULTS_DIR / "next_hour" / ds / "metrics_all.csv").exists():
        families.insert(0, "Next-hour LiteTCN")
        families.insert(1, "Next-hour earlier hybrid (reference)")
    if (RESULTS_DIR / "next_hour_hybrid" / ds / "metrics_all.csv").exists():
        families.insert(0, "Next-hour LiteTCN + local dynamics")
    if (RESULTS_DIR / "peak_next_hour" / ds / "metrics_all.csv").exists():
        families.insert(0, "Peak-calibrated next-hour adaptation")
    family = st.selectbox("Experiment", families)
    improved = family != "Original CARE-DA v2"
    transfer = family == "Selected using target history"
    strict = family in ("Strict label-free ensemble", "Strict shorter-context ensemble")
    context = family == "Strict shorter-context ensemble"
    peak = family in ("LiteTCN v3", "LiteTCN v3 + recent residuals")
    local = family == "LiteTCN v3 + recent residuals"
    residual = family == "LiteTCN-R (original encoder)"
    progressive = family == "LiteTCN-R gradual unfreezing"
    coupled = family == "LiteTCN predicted-driver decoder"
    next_hour = family in ("Next-hour LiteTCN", "Next-hour LiteTCN + local dynamics", "Next-hour earlier hybrid (reference)", "Peak-calibrated next-hour adaptation")
    next_hybrid = family == "Next-hour LiteTCN + local dynamics"
    next_reference = family == "Next-hour earlier hybrid (reference)"
    peak_next = family == "Peak-calibrated next-hour adaptation"
    event = family == "24-hour peak-aware boosted adaptation"
    meta, models = load_models(ds, improved, transfer, family == "DLinear transfer experiment", strict, context, peak, local, residual, progressive, coupled, next_hour, next_hybrid, event, next_reference, peak_next)
    if next_hour:
        model_label = "Peak-calibrated next-hour hybrid" if peak_next else "Earlier hybrid, first-hour output" if next_reference else "Next-hour LiteTCN + local dynamics" if next_hybrid else "Next-hour LiteTCN"
        st.info("Forecast horizon: 1 hour. Rolling evaluation refreshes the observed history at every hour.")
        if peak_next:
            st.caption("Observed trend selects a calibrated source/adapted expert mixture. Calibration uses peaks and dips in daily rolling episodes wholly inside the observed input histories; ordinary-error guards and a source fallback remain active.")
        elif next_reference:
            st.caption("Reference: first output of the previously trained 24-hour hybrid. Its weights are reused; this reference was not retrained as a next-hour model.")
        else:
            st.caption("The small source-trained TCN and its adapted model predict all variables. Adaptation withholds the final observed hour inside each target input; future target labels are excluded.")
    elif event:
        model_label = "LiteTCN + peak-aware boosted trajectory"
        st.info("Forecast horizon: 24 hours from a single origin.")
        st.caption("Source/target-input validation selects trajectory vs peak-geometry boosting and its mixture with LiteTCN. No actual future load/weather measurements are inputs.")
    elif coupled:
        model_label = "LiteTCN with source-selected temperature correction"
        decoder_control = meta["model_spec"]["control"].replace("_", " ")
        st.caption(f"Selected decoder control: {decoder_control}. The base remains multivariate; the temperature-only control masks additional driver conditioning. The target model uses the retained input-adapted backbone and a zero-correction fallback. Actual future drivers are never inputs.")
    elif progressive:
        model_label = "LiteTCN-R with gradual unfreezing"
        st.caption("Source validation compares head-only and gradual-unfreezing controls. Adaptation gradually unfreezes the encoder using withheld observations inside each input; the linear skip remains fixed.")
    elif residual:
        model_label = "LiteTCN-R (original encoder + regularized residual head)"
        st.caption("The shared linear skip starts from DLinear; the original source-trained LiteTCN encoder is retained. The residual can fall back to zero.")
    elif peak:
        model_label = "LiteTCN v3 (seasonal dilated TCN)"
        if local:
            model_label += " + input-only residual correction"
        st.caption(f"Source variant: {meta['model_spec']} · adaptation: {meta['adaptation']}")
    elif strict:
        model_label = "Channel-specific ridge + Patch MLP + DLinear / input-only adaptation"
        if context:
            st.caption(f"Linear branch uses the latest {meta['L'] - meta['H']} real input steps; its self-supervised task needs no padding.")
    elif improved:
        architecture = {"litetcn": "LiteTCN", "patch_mlp": "Patch MLP", "dlinear": "RevIN DLinear"}[meta["model_spec"]["architecture"]]
        adaptation = {"source_only": "source fallback", "replay_inner": "replay + inner forecast",
                      "replay_inner_mmd": "replay + inner forecast + MMD"}[meta["adaptation"]]
        model_label = f"{architecture} / {adaptation}"
    else:
        model_label = "CARE-DA v2"
    st.caption(f"Model: {model_label}")
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
            st.write("Raw OHLCV needs one extra row to compute the first return and overnight gap.")
        st.write(f"At least L = {meta['L']} rows (look-back); L + H = {meta['L'] + meta['H']} rows to "
                 "compare against actual values.")
        st.write("Upload one chronological series with complete values. Each row must use the training cadence: hourly for climate/energy, trading days for finance.")

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
    if getattr(main_model, "requires_calendar", False) and not isinstance(df.index, pd.DatetimeIndex):
        st.error("This model requires a date/timestamp column with consecutive hourly observations.")
        st.stop()
    if getattr(main_model, "requires_calendar", False) and not df.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        st.error("This model requires consecutive hourly rows; this CSV contains gaps or a different sampling interval.")
        st.stop()
    key_idx = cols.index(meta["key"])

    max_start = len(df) - L - (H if len(df) >= L + H else 0)
    c1, c2 = st.columns([3, 2])
    future_mode = c1.checkbox("Forecast after the final uploaded row", value=len(df) < L + H)
    if future_mode:
        start = len(df) - L
        c1.caption(f"Using rows {start} to {start + L - 1}; actual future values are unavailable.")
    elif max_start > 0:
        start = c1.slider("Start of look-back window (row index)", 0, max_start, max_start)
    else:
        start = 0
        c1.caption("One look-back window available.")
    default_variables = [meta["key"]] + [c for c in cols if c != meta["key"]][:3]
    show = c2.multiselect("Variables to plot", cols, default=default_variables)
    c3, c4 = st.columns([3, 2])
    n_show = c3.slider("Look-back steps to display", min(L, H), L, min(L, 48 if H == 1 else 4 * H),
                       help="The model always uses the full look-back window; this only changes the plot.")
    compare = is_tgt and c4.checkbox("Overlay source and earlier models", value=True)

    x = z[start:start + L][None]
    x_dates = df.index[start:start + L].to_numpy()[None]
    has_actual = start + L + H <= len(df)
    p_main = sc.inverse(forecast(main_model, x, x_dates)[0])
    p_src = sc.inverse(forecast(models["source_model"], x, x_dates)[0]) if compare else None
    p_v1 = None
    if compare and "adapted_model_v1" in models:
        p_v1 = sc.inverse(forecast(models["adapted_model_v1"], x, x_dates)[0])
    p_old = sc.inverse(forecast(models["legacy_adapted"], x, x_dates)[0][:H]) if compare and improved else None
    t_in = df.index[start + L - n_show:start + L]
    if has_actual:
        t_out = df.index[start + L:start + L + H]
    elif ds == "finance" and isinstance(df.index, pd.DatetimeIndex):
        t_out = pd.bdate_range(df.index[start + L - 1] + pd.Timedelta(days=1), periods=H)
        st.caption("Future finance dates follow weekdays; exchange holidays are not included in this calendar.")
    else:
        origin = df.index[start + L - 1]
        step = (origin - df.index[start + L - 2]) if L > 1 else 1
        t_out = [origin + step * (i + 1) for i in range(H)]

    fig = make_subplots(rows=len(show) or 1, cols=1, shared_xaxes=True, subplot_titles=show,
                        vertical_spacing=0.06)
    for r, c in enumerate(show, start=1):
        j = cols.index(c)
        first = r == 1
        fig.add_trace(go.Scatter(x=t_in, y=raw[start + L - n_show:start + L, j], name="Input (look-back)",
                                 line=dict(color=C_INPUT, width=2), showlegend=first, legendgroup="in"), r, 1)
        if has_actual:
            fig.add_trace(go.Scatter(x=t_out, y=raw[start + L:start + L + H, j], name="Actual",
                                     mode="lines+markers" if H == 1 else "lines", line=dict(color=C_ACTUAL, width=2), showlegend=first, legendgroup="act"), r, 1)
        if p_src is not None:
            fig.add_trace(go.Scatter(x=t_out, y=p_src[:, j], name="Source-only forecast",
                                     line=dict(color=C_SRC, width=2, dash="dash"), showlegend=first,
                                     legendgroup="src"), r, 1)
        if p_v1 is not None:
            fig.add_trace(go.Scatter(x=t_out, y=p_v1[:, j], name="CARE-DA v1 forecast",
                                     line=dict(color="#4a3aa7", width=2, dash="dot"), showlegend=first,
                                     legendgroup="v1"), r, 1)
        if p_old is not None:
            fig.add_trace(go.Scatter(x=t_out, y=p_old[:, j], name="Original CARE-DA v2",
                                     line=dict(color="#a765ab", width=2, dash="dot"),
                                     showlegend=first, legendgroup="old"), r, 1)
        fig.add_trace(go.Scatter(x=t_out, y=p_main[:, j],
                                 name=f"Adapted forecast ({model_label})" if is_tgt else "Source model forecast",
                                 mode="lines+markers" if H == 1 else "lines", line=dict(color=C_ADAPT, width=2), showlegend=first, legendgroup="main"), r, 1)
        fig.add_vline(x=t_out[0], line=dict(color="#c3c2b7", width=1, dash="dot"), row=r, col=1)
    fig.update_layout(height=230 * max(1, len(show)) + 60, hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", yref="container", y=0.995, yanchor="top", x=0),
                      margin=dict(l=10, r=10, t=75, b=10))
    st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------ metrics
    st.subheader("Goodness of forecast")
    if has_actual:
        y = z[start + L:start + L + H][None]
        m_win = all_metrics(x, y, forecast(main_model, x, x_dates), meta["season"], sc, key_idx)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Window MSE (z-scored)", f"{m_win['MSE']:.3f}")
        k2.metric("Window MAE (z-scored)", f"{m_win['MAE']:.3f}")
        k3.metric("Window-scaled MAE", f"{m_win['MASE']:.3f}", help="MAE divided by historical seasonal differences in this input window; use the rolling table for direct baseline comparisons.")
        k4.metric(f"{meta['key']} RMSE", f"{m_win['key_RMSE']:.3f}", help=meta["key_unit"])
    else:
        st.info("Forecast extends beyond the file - no actual values to score this window.")

    n_win = (len(df) - L - H) // H + 1 if len(df) >= L + H else 0
    if n_win >= 1:
        idx = [i * H for i in range(n_win)]
        X = np.stack([z[i:i + L] for i in idx])
        X_dates = np.stack([df.index[i:i + L].to_numpy() for i in idx])
        Y = np.stack([z[i + L:i + L + H] for i in idx])
        rows = {"Selected adapted model" if is_tgt else "Source model": forecast(main_model, X, X_dates)}
        if is_tgt:
            rows["Source-only (no adaptation)"] = forecast(models["source_model"], X, X_dates)
            if "adapted_model_v1" in models:
                rows["CARE-DA v1"] = forecast(models["adapted_model_v1"], X, X_dates)
        if improved:
            old_key = "legacy_adapted" if is_tgt else "legacy_source"
            rows["Original CARE-DA v2" if is_tgt else "Original source model"] = forecast(models[old_key], X, X_dates)[:, :H]
        rows["Seasonal naive"] = seasonal_naive(X, H, meta["season"])
        rows["Persistence"] = seasonal_naive(X, H, 1)
        table = pd.DataFrame({k: all_metrics(X, Y, v, meta["season"], sc, key_idx) for k, v in rows.items()}).T
        table = table.rename(columns={"key_RMSE": f"{meta['key']} RMSE", "key_MAE": f"{meta['key']} MAE",
                                      "MASE": "Window-scaled MAE"})
        st.markdown(f"**Whole-file rolling evaluation** — {n_win} non-overlapping windows (L={L}, H={H})")
        st.dataframe(table.style.format("{:.4f}").highlight_min(axis=0, color="#d6e6f9"), width="stretch")
        if is_tgt:
            a, s = table.iloc[0]["MSE"], table.iloc[1]["MSE"]
            message = f"Adaptation changes MSE by {100 * (a - s) / s:+.1f}% relative to the un-adapted model."
            (st.success if a < s else st.warning)(message)
        per_variable = pd.DataFrame({
            "RMSE (original units)": np.sqrt((((rows[next(iter(rows))] - Y) * sc.std) ** 2).mean((0, 1))),
            "MAE (original units)": (np.abs(rows[next(iter(rows))] - Y) * sc.std).mean((0, 1)),
        }, index=cols)
        st.dataframe(per_variable.style.format("{:.4f}"), width="stretch")
        if H == 1:
            st.subheader("Rolling next-hour forecasts")
            st.caption("Each point forecasts the next hour using observations available before that hour. The history is updated hourly, and the model weights stay fixed during this evaluation.")
            rolling_time = df.index[L:L+n_win]
            rolling_fig = make_subplots(rows=len(show) or 1, cols=1, shared_xaxes=True, subplot_titles=show, vertical_spacing=.06)
            main_name = next(iter(rows))
            for r, column in enumerate(show, start=1):
                j = cols.index(column)
                rolling_fig.add_trace(go.Scatter(x=rolling_time,y=raw[L:L+n_win,j],name="Actual",line=dict(color=C_ACTUAL,width=2),showlegend=r==1),r,1)
                rolling_fig.add_trace(go.Scatter(x=rolling_time,y=rows[main_name][:,0,j]*sc.std[j]+sc.mean[j],name="Rolling next-hour forecast",line=dict(color=C_ADAPT,width=2),showlegend=r==1),r,1)
                if is_tgt:
                    rolling_fig.add_trace(go.Scatter(x=rolling_time,y=rows["Source-only (no adaptation)"][:,0,j]*sc.std[j]+sc.mean[j],name="Source-only next-hour forecast",line=dict(color=C_SRC,width=1,dash="dot"),showlegend=r==1),r,1)
            rolling_fig.update_layout(height=230*max(1,len(show))+60,template="plotly_white",hovermode="x unified",legend=dict(orientation="h"),margin=dict(l=10,r=10,t=60,b=10))
            st.plotly_chart(rolling_fig,width="stretch")
            rolling_out=pd.DataFrame(sc.inverse(rows[main_name][:,0]),columns=cols,index=pd.Index(rolling_time,name="date"))
            st.download_button("Download rolling next-hour forecasts",rolling_out.to_csv().encode(),f"rolling_next_hour_{ds}_{role}.csv","text/csv")

    out = pd.DataFrame(p_main, columns=cols, index=pd.Index(t_out, name="date"))
    st.download_button("Download forecast CSV", out.to_csv().encode(),
                       f"forecast_{ds}_{role}_{meta.get('experiment', 'legacy')}.csv", "text/csv")

with tab_bench:
    for short_family, label in (("peak_next_hour", "Peak-calibrated next-hour adaptation"), ("next_hour_hybrid", "Next-hour LiteTCN + local dynamics"), ("next_hour", "Next-hour LiteTCN")):
        short_results = RESULTS_DIR / short_family / ds / "metrics_all.csv"
        if short_results.exists():
            st.markdown(f"**{label} — H = 1 hour**. Every rolling prediction uses observations refreshed at its own origin. Compare these metrics with next-hour baselines; the 24-hour experiment tables below use a different horizon.")
            st.dataframe(pd.read_csv(short_results).groupby(["domain", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index().round(4), width="stretch")
            short_peaks = RESULTS_DIR / short_family / ds / "rolling_peak_metrics.csv"
            st.dataframe(pd.read_csv(short_peaks).groupby(["domain", "method"])[["RMSE", "MAE_at_actual_peak", "MAE_at_actual_trough", "ramp_MAE"]].mean().reset_index().round(4), width="stretch")
    event_results = RESULTS_DIR / "event_boost" / ds / "metrics_all.csv"
    if event_results.exists():
        st.markdown("**Peak-aware boosted adaptation — H = 24 hours from one origin**. Geometry training predicts trajectory coefficients, peak/trough height and timing. Source validation and target-input pretexts select its calibration and mixture with LiteTCN.")
        st.dataframe(pd.read_csv(event_results).groupby(["domain", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index().round(4), width="stretch")
    final_report = ROOT / "report" / "IE643_Forecast_Horizons_Results.html"
    if final_report.exists():
        st.download_button("Download next-hour and 24-hour results", final_report.read_bytes(), final_report.name, "text/html")
    residual_results = RESULTS_DIR / "residual_litetcn" / ds / "metrics_all.csv"
    if residual_results.exists():
        st.markdown("**LiteTCN-R: original encoder retained** — the linear skip is initialized to the DLinear source forecast. Regularized residual-head training and input-only adaptation include an explicit zero-residual baseline. A lower error for `linear_skip_only` means the neural residual did not help on this development test.")
        table = pd.read_csv(residual_results).groupby(["domain", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index()
        st.dataframe(table.round(4), width="stretch")
    local_results = RESULTS_DIR / "peak_calibrated" / ds / "metrics_all.csv"
    if local_results.exists():
        st.markdown("**LiteTCN with recent forecast residuals** — correction uses observations entirely inside each input window. Source-validation and nested target-input validation select its strength; no target forecasting labels are used.")
        table = pd.read_csv(local_results).groupby(["domain", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index()
        st.dataframe(table.round(4), width="stretch")
    peak_results = RESULTS_DIR / "peak_litetcn" / ds / "metrics_all.csv"
    if peak_results.exists():
        st.markdown("**LiteTCN v3: retained TCN backbone** — seasonal folding, channel-specific linear skip and source-selected calendar/shape-loss ablations. Adaptation forecasts withheld values inside each target input; target forecasting labels are excluded from training and selection.")
        table = pd.read_csv(peak_results).groupby(["domain", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index()
        st.dataframe(table.round(4), width="stretch")
        peaks = RESULTS_DIR / "peak_litetcn" / ds / "peak_metrics.csv"
        st.dataframe(pd.read_csv(peaks).groupby(["domain", "method"])[["RMSE", "MAE_at_actual_peak", "MAE_at_actual_trough", "ramp_MAE"]].mean().reset_index().round(4), width="stretch")
        with st.expander("LiteTCN v3 selection record"):
            st.json(json.loads((MODEL_DIR / "peak_litetcn" / ds / "selection.json").read_text()))
    progressive_results = RESULTS_DIR / "progressive_litetcn" / ds / "metrics_all.csv"
    if progressive_results.exists():
        st.markdown("**Gradual LiteTCN-R unfreezing** — matched head-only control, source-validation selection and target-input-only adaptation. Test results use the previously inspected development period.")
        st.dataframe(pd.read_csv(progressive_results).groupby(["domain", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index().round(4), width="stretch")
        with st.expander("Gradual-unfreezing selection record"):
            st.json(json.loads((MODEL_DIR / "progressive_litetcn" / ds / "selection.json").read_text()))
    progressive_report = ROOT / "report" / "IE643_Progressive_LiteTCN_Results.html"
    if progressive_report.exists():
        st.download_button("Download gradual-unfreezing results and plots", progressive_report.read_bytes(), progressive_report.name, "text/html")
    coupled_results = RESULTS_DIR / "coupled_litetcn" / ds / "metrics_all.csv"
    if coupled_results.exists():
        st.markdown("**Predicted-driver decoder** — compare temperature-only and driver-conditioned corrections to the same frozen multivariate LiteTCN. Source validation selects the control; adaptation uses target input tails only. Test results are development evidence.")
        st.dataframe(pd.read_csv(coupled_results).groupby(["domain", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index().round(4), width="stretch")
        with st.expander("Predicted-driver decoder selection record"):
            st.json(json.loads((MODEL_DIR / "coupled_litetcn" / ds / "selection.json").read_text()))
    coupled_report = ROOT / "report" / "IE643_Coupled_LiteTCN_Results.html"
    coupled_transfer_results = RESULTS_DIR / "coupled_transfer" / ds / "metrics_all.csv"
    if coupled_transfer_results.exists():
        st.markdown("**Decoder on retained adapted backbone** — direct comparison against the same backbone with zero correction. Only temperature outputs can change. Selection uses source validation and target-input proxies; the source-backbone-only control above is retained as a negative result.")
        st.dataframe(pd.read_csv(coupled_transfer_results).groupby(["domain", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index().round(4), width="stretch")
    if coupled_report.exists():
        st.download_button("Download predicted-driver decoder results and plots", coupled_report.read_bytes(), coupled_report.name, "text/html")
    peak_report = ROOT / "report" / "IE643_LiteTCN_Model_Results.html"
    if peak_report.exists():
        st.download_button("Download LiteTCN experiment notes and plots", peak_report.read_bytes(), peak_report.name, "text/html")
    strict_report = ROOT / "report" / "IE643_Strict_Refinement_Report.html"
    if strict_report.exists():
        st.download_button("Download latest label-free report", strict_report.read_bytes(), strict_report.name, "text/html")
    strict_results = RESULTS_DIR / "strict_refined" / ds / "metrics_all.csv"
    if strict_results.exists():
        st.markdown("**Strict label-free refinement** — source labels and forecasting withheld values inside each target input window. No target forecasting labels or cross-window label reconstruction are used in training or selection. Results reuse the development test period.")
        current = pd.read_csv(strict_results)
        current = current.groupby(["domain", "method"]).agg(MSE_mean=("MSE", "mean"), MSE_seed_SD=("MSE", "std"),
                     key_RMSE_mean=("key_RMSE", "mean"), MAE_mean=("MAE", "mean"), MASE_train_mean=("MASE_train", "mean")).reset_index()
        st.dataframe(current.round(4), width="stretch")
        with st.expander("Strict selection record"):
            st.json(json.loads((MODEL_DIR / "strict_refined" / ds / "selection.json").read_text()))
    context_results = RESULTS_DIR / "strict_context" / ds / "metrics_all.csv"
    if context_results.exists():
        st.markdown("**Shorter-context follow-up** — the linear branch uses L-H real steps in source training and within-input self-supervision. Exploratory results on the same development test; no target forecasting labels used.")
        table = pd.read_csv(context_results).groupby(["domain", "method"])[["MSE", "key_RMSE", "MASE_train"]].mean().reset_index()
        st.dataframe(table.round(4), width="stretch")
    report_file = ROOT / "report" / "IE643_Improvement_Report.html"
    if report_file.exists():
        st.download_button("Download stage report", report_file.read_bytes(), report_file.name, "text/html")
    new_summary = RESULTS_DIR / "improved" / ds / "summary.csv"
    if new_summary.exists():
        st.markdown("**Architecture and adaptation experiments** — held-out test evaluation after validation selection. `supervised_reference` uses target labels. Original seed-0 models are fixed historical references, repeated across seeds.")
        st.dataframe(pd.read_csv(new_summary).round(4), width="stretch")
        selection_file = MODEL_DIR / "improved" / ds / "selection.json"
        if selection_file.exists():
            with st.expander("Model selection record"):
                st.json(json.loads(selection_file.read_text()))
    transfer_summary = RESULTS_DIR / "transfer_selection" / ds / "summary.csv"
    if transfer_summary.exists():
        st.markdown("**Follow-up using target history for model selection** — compares the initial source choice with DLinear adaptation. Selection uses source validation and observed target inputs. This exploratory extension reuses the development test period.")
        st.dataframe(pd.read_csv(transfer_summary).round(4), width="stretch")
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
