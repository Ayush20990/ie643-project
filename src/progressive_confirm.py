"""Frozen-policy check on newly downloaded 1-17 October 2026 weather.

This short follow-up was reserved before training ended. It is never used for
training, selecting checkpoints, or selecting policies. Overlapping hourly
forecast windows are not independent experimental repetitions.
"""
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
from .peak_model import PeakLiteTCN, pack_calendar
from .progressive_tcn import ProgressiveLiteTCN
from .progressive_train import sha
from .train import predict


def main():
    torch.set_num_threads(2)
    cfg = DATASETS["climate"]
    out = RESULTS_DIR/"progressive_litetcn"/"climate_oct2026"
    cache = PROC_DIR.parent/"confirmatory_oct2026"/"climate"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    # Both datasets must finish checkpoint/policy selection before this download.
    for ds in ("climate", "energy"):
        assert (RESULTS_DIR/"progressive_litetcn"/ds/"selection_frozen.json").exists()
    folders = {k: MODEL_DIR/k/"climate" for k in ("progressive_litetcn", "peak_litetcn", "strict_context")}
    paths = [p for d in folders.values() for p in sorted(d.rglob("*.pt"))]
    paths += [d/"meta.json" for d in folders.values()]
    paths += [MODEL_DIR/"climate"/f"{role}_model.pt" for role in ("source", "adapted")]
    freeze = {str(p.relative_to(MODEL_DIR)): sha(p) for p in paths}
    freeze_path = out/"frozen_before_download.json"
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text())["model_sha256"] == freeze
    else:
        write_json(freeze_path, {"model_sha256": freeze, "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
             "test_period": "2026-10-01 through 2026-10-17", "new_model_selection_on_this_period": False})
    metadata = {k: json.loads((d/"meta.json").read_text()) for k,d in folders.items()}
    rows, peaks, horizons, examples, provenance = [], [], [], {}, {}
    for role in ("source", "target"):
        site = cfg[role]
        params = {**CLIMATE_SITES[site], "start_date": "2026-09-24", "end_date": "2026-10-17",
                  "hourly": ",".join(cfg["columns"]), "timezone": "Asia/Kolkata"}
        path = cache/f"{site}.json"
        if not path.exists():
            response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=45)
            response.raise_for_status()
            path.write_text(response.text, encoding="utf-8")
        js = json.loads(path.read_text(encoding="utf-8"))
        reference_path = PROC_DIR.parent/"confirmatory_2026"/"climate"/f"{site}.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        assert js["hourly_units"] == reference["hourly_units"]
        frame = pd.DataFrame(js["hourly"]).rename(columns={"time": "date"})
        frame["date"] = pd.to_datetime(frame.date)
        values = frame[cfg["columns"]].to_numpy(dtype=np.float32)
        assert np.isfinite(values).all()
        assert frame.date.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
        frame.to_csv(cache/f"{site}.csv", index=False)
        scaler = Scaler(**metadata["progressive_litetcn"][f"{role}_scaler"])
        x, y = make_windows([scaler.transform(values)], cfg["L"], cfg["H"])
        assert frame.date.iloc[cfg["L"]] == pd.Timestamp("2026-10-01")
        dates = np.lib.stride_tricks.sliding_window_view(frame.date.to_numpy(), cfg["L"])[:len(x)]
        packed = torch.from_numpy(pack_calendar(x.numpy(), dates))
        dom = {"test": (x,y), "scaler": scaler}
        key = cfg["columns"].index(cfg["key"])
        denom = fixed_scale("climate", role, scaler)
        checkpoint = "source_model" if role=="source" else "adapted_model"
        old = pd.read_csv(reference_path.with_suffix(".csv"), parse_dates=["date"])
        overlap = old.merge(frame, on="date", suffixes=("_old", "_new"))
        provenance[role] = {"request": params, "data_sha256": sha(path), "windows": len(x),
            "forecast_start": str(frame.date.iloc[cfg["L"]]), "last_actual": str(frame.date.iloc[-1]),
            "hourly_units": js["hourly_units"],
            "overlap_RMSE": {c: float(np.sqrt(np.mean((overlap[c+"_old"]-overlap[c+"_new"])**2))) for c in cfg["columns"]}}
        for seed in (0,1,2):
            progressive = ProgressiveLiteTCN(cfg)
            progressive.load_state_dict(torch.load(folders["progressive_litetcn"]/f"seed_{seed}"/f"{checkpoint}.pt", weights_only=True))
            spec = {k:v for k,v in metadata["peak_litetcn"]["model_spec"].items() if k not in ("architecture","shape_loss")}
            previous_tcn = PeakLiteTCN(cfg, **spec)
            previous_tcn.load_state_dict(torch.load(folders["peak_litetcn"]/f"seed_{seed}"/f"{checkpoint}.pt", weights_only=True))
            hybrid = ContextEnsemble(cfg, metadata["strict_context"]["model_spec"]["neural_spec"])
            hybrid.load_state_dict(torch.load(folders["strict_context"]/f"seed_{seed}"/f"{checkpoint}.pt", weights_only=True))
            legacy = make_forecaster(cfg)
            legacy.load_state_dict(torch.load(MODEL_DIR/"climate"/f"{checkpoint}.pt", weights_only=True))
            for name, model, inputs in (("progressive_LiteTCN_R", progressive, x), ("previous_LiteTCN_v3", previous_tcn, packed),
                                        ("previous_hybrid", hybrid, x), ("original_LiteTCN_seed0", legacy, x)):
                prediction = predict(model, inputs, bs=256)
                scores, _ = evaluate("climate", cfg, prediction, dom, denom)
                rows.append({"domain": role, "seed": seed, "method": name, **scores})
                peaks.append({"domain": role, "seed": seed, "method": name,
                              **peak_metrics(y.numpy()[...,key]*scaler.std[key], prediction[...,key]*scaler.std[key])})
                for h in (1,3,6,12,24):
                    horizons.append({"domain": role, "seed": seed, "method": name, "horizon_hours": h,
                        "key_RMSE": float(np.sqrt(np.mean((prediction[:,:h,key]-y.numpy()[:,:h,key])**2))*scaler.std[key])})
                if seed==0:
                    ix = np.linspace(0,len(x)-1,min(256,len(x)),dtype=int)
                    examples[f"{role}_{name}"] = prediction[ix]
                    examples[f"{role}_X"], examples[f"{role}_Y"] = x.numpy()[ix], y.numpy()[ix]
                    examples[f"{role}_indices"] = ix
        print(f"New October climate {role} evaluation complete: {len(x)} origins", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"metrics_all.csv", index=False)
    pd.DataFrame(peaks).to_csv(out/"peak_metrics.csv", index=False)
    pd.DataFrame(horizons).to_csv(out/"horizon_metrics.csv", index=False)
    np.savez_compressed(out/"forecast_examples.npz", **examples)
    write_json(out/"provenance.json", provenance)
    assert all(sha(MODEL_DIR/p)==checksum for p, checksum in freeze.items())
    print(frame.groupby(["domain","method"])[["MSE","key_RMSE"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
