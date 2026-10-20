"""One frozen-model evaluation on new Jan-Sep 2026 climate observations.

No training, calibration-policy selection or model selection occurs in this file.
Per-window residual correction, where enabled, uses observed input values only.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import requests
import torch

from .config import CLIMATE_SITES, DATASETS, MODEL_DIR, PROC_DIR, RESULTS_DIR
from .context_models import ContextEnsemble
from .data import Scaler, make_windows
from .diagnose_peaks import peak_metrics
from .improve import evaluate, fixed_scale, write_json
from .model import make_forecaster
from .peak_calibration import CalibratedLiteTCN
from .peak_model import pack_calendar
from .peak_train import make_peak
from .train import predict


def main():
    torch.set_num_threads(2)
    cfg = DATASETS["climate"]
    out = RESULTS_DIR/"peak_litetcn"/"climate_2026"
    cache = PROC_DIR.parent/"confirmatory_2026"/"climate"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    folders = {name: MODEL_DIR/name/"climate" for name in ("peak_litetcn", "peak_calibrated", "strict_context")}
    checkpoint_files = [p for folder in folders.values() for p in folder.glob("seed_*/*_model.pt")]
    checkpoint_files += [folder/"meta.json" for folder in folders.values()]
    checkpoint_files += [MODEL_DIR/"climate"/f"{role}_model.pt" for role in ("source", "adapted")]
    frozen = {str(p.relative_to(MODEL_DIR)): hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_files}
    freeze_path = out/"frozen_before_download.json"
    if freeze_path.exists():
        if json.loads(freeze_path.read_text())["model_sha256"] != frozen:
            raise ValueError("Confirmatory model files changed")
    else:
        write_json(freeze_path, {"model_sha256": frozen, "utc_time": pd.Timestamp.now(tz="UTC").isoformat(),
                   "no_further_training_or_selection_on_this_period": True})
    metadata = {name: json.loads((p/"meta.json").read_text()) for name, p in folders.items()}
    rows, peaks, horizons, provenance = [], [], [], {}
    for role in ("source", "target"):
        site = cfg[role]
        params = {**CLIMATE_SITES[site], "start_date": "2025-12-25", "end_date": "2026-09-30",
                  "hourly": ",".join(cfg["columns"]), "timezone": "Asia/Kolkata"}
        path = cache/f"{site}.json"
        if not path.exists():
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=55)
            r.raise_for_status()
            path.write_text(r.text, encoding="utf-8")
        js = json.loads(path.read_text(encoding="utf-8"))
        reference_path = PROC_DIR.parent/"confirmatory_2025"/"climate"/f"{site}.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if js["hourly_units"] != reference["hourly_units"]:
            raise ValueError("New weather units differ")
        frame = pd.DataFrame(js["hourly"]).rename(columns={"time": "date"})
        frame["date"] = pd.to_datetime(frame.date)
        values = frame[cfg["columns"]].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or not frame.date.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError("New period must be finite and hourly; no interpolation")
        frame.to_csv(cache/f"{site}.csv", index=False)
        scaler = Scaler(**metadata["peak_litetcn"][f"{role}_scaler"])
        x, y = make_windows([scaler.transform(values)], cfg["L"], cfg["H"])
        if frame.date.iloc[cfg["L"]] != pd.Timestamp("2026-01-01"):
            raise ValueError("Unexpected confirmatory start")
        dates = np.lib.stride_tricks.sliding_window_view(frame.date.to_numpy(), cfg["L"])[:len(x)]
        packed = torch.from_numpy(pack_calendar(x.numpy(), dates))
        dom = {"test": (x, y), "scaler": scaler}
        denominator = fixed_scale("climate", role, scaler)
        old = pd.read_csv(reference_path.with_suffix(".csv"), parse_dates=["date"])
        overlap = old.merge(frame, on="date", suffixes=("_old", "_new"))
        provenance[role] = {"request": params, "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "windows": len(x), "hourly_units": js["hourly_units"],
            "overlap_2025_RMSE": {c: float(np.sqrt(np.mean((overlap[c+"_old"]-overlap[c+"_new"])**2))) for c in cfg["columns"]}}
        checkpoint = "source_model" if role == "source" else "adapted_model"
        key = cfg["columns"].index(cfg["key"])
        for seed in (0, 1, 2):
            base = make_peak(cfg, metadata["peak_litetcn"]["model_spec"])
            base.load_state_dict(torch.load(folders["peak_litetcn"]/f"seed_{seed}"/f"{checkpoint}.pt", weights_only=True))
            local_meta = json.loads((RESULTS_DIR/"peak_calibrated"/"climate"/f"seed_{seed}_selection.json").read_text())
            config = local_meta["calibration_source" if role == "source" else "calibration_target"]
            local = CalibratedLiteTCN(make_peak(cfg, metadata["peak_litetcn"]["model_spec"]), **config)
            local.load_state_dict(torch.load(folders["peak_calibrated"]/f"seed_{seed}"/f"{checkpoint}.pt", weights_only=True))
            hybrid = ContextEnsemble(cfg, metadata["strict_context"]["model_spec"]["neural_spec"])
            hybrid.load_state_dict(torch.load(folders["strict_context"]/f"seed_{seed}"/f"{checkpoint}.pt", weights_only=True))
            legacy = make_forecaster(cfg)
            legacy.load_state_dict(torch.load(MODEL_DIR/"climate"/f"{checkpoint}.pt", weights_only=True))
            methods = {"LiteTCN_v3": (base, packed), "LiteTCN_v3_local": (local, packed),
                       "previous_hybrid": (hybrid, x), "original_LiteTCN_seed0": (legacy, x)}
            for name, (model, inputs) in methods.items():
                prediction = predict(model, inputs, bs=256)
                scores, _ = evaluate("climate", cfg, prediction, dom, denominator)
                rows.append({"dataset": "climate", "period": "2026_Jan_Sep", "domain": role, "seed": seed, "method": name, **scores})
                peaks.append({"domain": role, "seed": seed, "method": name,
                    **peak_metrics(y.numpy()[..., key]*scaler.std[key], prediction[..., key]*scaler.std[key])})
                for h in (1, 3, 6, 12, 24):
                    horizons.append({"domain": role, "seed": seed, "method": name, "horizon_hours": h,
                        "temperature_RMSE": float(np.sqrt(np.mean((prediction[:, :h, key]-y.numpy()[:, :h, key])**2))*scaler.std[key])})
        print(f"2026 {role} evaluated with frozen policies", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"metrics_all.csv", index=False)
    pd.DataFrame(peaks).to_csv(out/"peak_metrics.csv", index=False)
    pd.DataFrame(horizons).to_csv(out/"horizon_metrics.csv", index=False)
    write_json(out/"provenance.json", provenance)
    print(frame.groupby(["domain", "method"])[["MSE", "key_RMSE"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
