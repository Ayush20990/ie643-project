"""Shared checkpoint loading and strict, causal CSV validation for the interface."""
import json

import numpy as np
import pandas as pd
import torch

from .config import MODEL_DIR
from .data_download import ohlcv_to_features
from .model import make_forecaster


def load_bundle(ds, improved=False, transfer=False):
    folder = (MODEL_DIR / "transfer_selection" / ds if transfer else
              MODEL_DIR / "improved" / ds if improved else MODEL_DIR / ds)
    meta = json.loads((folder / "meta.json").read_text())
    models = {}
    for name in ("source_model", "adapted_model", "adapted_model_v1"):
        path = folder / f"{name}.pt"
        if not path.exists():
            continue
        model = make_forecaster(meta, meta.get("model_spec"))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models[name] = model.eval()
    return meta, models


def load_dlinear_bundle(ds):
    """Expose the DLinear family, with its adapter chosen by the input-only proxy."""
    folder = MODEL_DIR / "transfer_selection" / ds
    meta = json.loads((folder / "meta.json").read_text())
    selection = json.loads((folder / "selection.json").read_text())
    eligible = [name for name, score in selection["scores"].items()
                if name.startswith("dlinear_") and score["eligible"]]
    chosen = min(eligible, key=lambda name: selection["scores"][name]["proxy"]) if eligible else "dlinear_source"
    seed = meta["training_seed"]
    source_path = MODEL_DIR / "improved" / ds / f"seed_{seed}" / "dlinear.pt"
    recipe = "source_only" if chosen == "dlinear_source" else chosen.removeprefix("dlinear_")
    adapted_path = source_path if recipe == "source_only" else folder / f"seed_{seed}" / f"{recipe}.pt"
    meta.update(model_spec={"architecture": "dlinear"}, adaptation=recipe,
                experiment="dlinear_transfer", _checkpoint_path=str(adapted_path))
    models = {}
    for name, path in (("source_model", source_path), ("adapted_model", adapted_path)):
        model = make_forecaster(meta, meta["model_spec"])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models[name] = model.eval()
    return meta, models


def load_strict_bundle(ds, context=False):
    from .strict_models import StrictEnsemble
    from .context_models import ContextEnsemble
    folder = MODEL_DIR / ("strict_context" if context else "strict_refined") / ds
    meta = json.loads((folder / "meta.json").read_text())
    models = {}
    for name in ("source_model", "adapted_model"):
        model = (ContextEnsemble if context else StrictEnsemble)(meta, meta["model_spec"]["neural_spec"])
        model.load_state_dict(torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True))
        models[name] = model.eval()
    return meta, models


def load_peak_bundle(ds, calibrated=False):
    from .peak_model import PeakLiteTCN
    from .peak_calibration import CalibratedLiteTCN
    folder = MODEL_DIR / ("peak_calibrated" if calibrated else "peak_litetcn") / ds
    meta = json.loads((folder / "meta.json").read_text())
    spec = {k: v for k, v in meta["model_spec"].items() if k not in ("architecture", "shape_loss")}
    models = {}
    for name in ("source_model", "adapted_model"):
        model = PeakLiteTCN(meta, **spec)
        if calibrated:
            config = meta["calibration_source" if name == "source_model" else "calibration_target"]
            model = CalibratedLiteTCN(model, **config)
        model.load_state_dict(torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True))
        models[name] = model.eval()
    return meta, models


def load_residual_bundle(ds):
    from .residual_tcn import ResidualLiteTCN
    folder = MODEL_DIR / "residual_litetcn" / ds
    meta = json.loads((folder / "meta.json").read_text())
    models = {}
    for name in ("source_model", "adapted_model"):
        model = ResidualLiteTCN(meta)
        model.load_state_dict(torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True))
        models[name] = model.eval()
    return meta, models


def load_progressive_bundle(ds):
    from .progressive_tcn import ProgressiveLiteTCN
    folder = MODEL_DIR / "progressive_litetcn" / ds
    meta = json.loads((folder / "meta.json").read_text())
    models = {}
    for name in ("source_model", "adapted_model"):
        model = ProgressiveLiteTCN(meta)
        model.load_state_dict(torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True))
        models[name] = model.eval()
    return meta, models


def load_coupled_bundle(ds, transfer=False):
    from .coupled_tcn import CoupledLiteTCN
    folder = MODEL_DIR / ("coupled_transfer" if transfer else "coupled_litetcn") / ds
    meta = json.loads((folder / "meta.json").read_text())
    spec = {k:v for k,v in meta["model_spec"].items() if k != "architecture"}
    models = {}
    for name in ("source_model", "adapted_model"):
        model = CoupledLiteTCN(meta, **spec)
        model.load_state_dict(torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True))
        models[name] = model.eval()
    return meta, models


def load_next_hour_bundle(ds, hybrid=False):
    from .nowcast_tcn import NextHourTCN
    from .nowcast_hybrid import NextHourHybrid
    folder = MODEL_DIR / ("next_hour_hybrid" if hybrid else "next_hour") / ds
    meta = json.loads((folder / "meta.json").read_text())
    models = {}
    for name in ("source_model", "adapted_model"):
        model = (NextHourHybrid if hybrid else NextHourTCN)(meta)
        model.load_state_dict(torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True))
        models[name] = model.eval()
    return meta, models


def load_next_hour_reference_bundle(ds):
    from .horizon_reference import FirstHourReference
    meta, models = load_strict_bundle(ds, context=ds == "climate")
    meta.update(H=1, experiment="first_hour_reference", adaptation="existing_input_only_hybrid",
                model_spec={"architecture":"first_hour_reference"},
                forecast_semantics="first-hour output of the existing 24-hour model, refreshed observations each hour")
    return meta, {name: FirstHourReference(model).eval() for name, model in models.items()}


def load_peak_next_hour_bundle(ds):
    from .peak_nowcast import PeakNowcast
    folder = MODEL_DIR / "peak_next_hour" / ds
    meta = json.loads((folder / "meta.json").read_text())
    spec = meta["model_spec"]
    models = {}
    for name in ("source_model", "adapted_model"):
        model = PeakNowcast(meta, spec["reference_kind"], spec["neural_spec"])
        model.load_state_dict(torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True))
        models[name] = model.eval()
    return meta, models


def load_event_boost_bundle(ds):
    from catboost import CatBoostRegressor
    from .event_boost import EventBoostForecaster
    folder = MODEL_DIR / "event_boost" / ds
    meta = json.loads((folder / "meta.json").read_text())
    spec = meta["model_spec"]
    models = {}
    for role in ("source", "adapted"):
        choice = meta["selection"][role]
        tree = None
        if choice["weight"]:
            tree = CatBoostRegressor()
            tree.load_model(str(folder / "seed_0" / f"{role}_tree.cbm"))
        model = EventBoostForecaster(meta, spec["base_kind"], spec["base_spec"], tree, choice)
        model.base.load_state_dict(torch.load(folder / "seed_0" / f"{role}_backbone.pt", map_location="cpu", weights_only=True))
        models[f"{role}_model"] = model.eval()
    return meta, models


def read_forecast_csv(file, ds, columns):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("CSV contains duplicate column names.")
    if "series" in df and df["series"].nunique(dropna=False) > 1:
        raise ValueError("Upload one series at a time; this CSV contains multiple series.")
    date_col = next((c for c in df if c.lower() in ("date", "time", "datetime", "timestamp")), None)
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="raise")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Dates must be valid, unique and in increasing chronological order.")
    if ds == "finance" and not set(columns).issubset(df.columns):
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            if date_col is None:
                raise ValueError("Raw OHLCV data needs a date column.")
            df = ohlcv_to_features(df.rename(columns={date_col: "Date"}))
            date_col = "date"
    missing = [c for c in columns if c not in df]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    out = df[columns].astype(float)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Forecast variables must be finite with no missing values. Fill gaps using past observations before uploading.")
    out.index = pd.DatetimeIndex(pd.to_datetime(df[date_col])) if date_col else pd.RangeIndex(len(df))
    return out


@torch.no_grad()
def forecast_batches(model, xs, batch_size=256, dates=None):
    model.eval()
    x = torch.as_tensor(xs, dtype=torch.float32)
    if getattr(model, "requires_calendar", False) and x.shape[-1] == model.C:
        from .peak_model import pack_calendar
        if dates is None:
            raise ValueError("This calendar-aware LiteTCN needs a date/timestamp column for calendar features")
        dates = np.asarray(dates)
        if dates.shape != tuple(x.shape[:2]):
            raise ValueError("Timestamp windows must match the input windows")
        index = pd.DatetimeIndex(dates.reshape(-1))
        if index.tz is not None:
            index = index.tz_localize(None)  # Preserve dataset-local wall-clock convention.
        times = index.to_numpy(dtype="datetime64[ns]").reshape(dates.shape)
        if not np.all(np.diff(times, axis=1) == np.timedelta64(1, "h")):
            raise ValueError("Calendar-aware climate/energy forecasts require consecutive hourly rows")
        x = torch.from_numpy(pack_calendar(x.numpy(), times))
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]).numpy()
