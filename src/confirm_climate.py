"""Evaluate frozen models on a new 2025 climate period; never train or select here."""
import hashlib
import json

import numpy as np
import pandas as pd
import requests
import torch

from .config import CLIMATE_SITES, DATASETS, MODEL_DIR, PROC_DIR, RAW_DIR, RESULTS_DIR
from .data import Scaler, make_windows
from .improve import evaluate, fixed_scale, write_json
from .metrics import seasonal_naive
from .model import make_forecaster
from .strict_models import StrictEnsemble


def main():
    torch.set_num_threads(4)
    cfg = DATASETS["climate"]
    out = RESULTS_DIR / "strict_refined" / "confirmatory_2025"
    cache = PROC_DIR.parent / "confirmatory_2025" / "climate"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    folder = MODEL_DIR / "strict_refined" / "climate"
    meta = json.loads((folder / "meta.json").read_text())
    frozen = {str(p.relative_to(MODEL_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(folder.glob("seed_*/*_model.pt"))}
    freeze_path = out / "freeze_before_download.json"
    if freeze_path.exists():
        if json.loads(freeze_path.read_text())["model_sha256"] != frozen:
            raise ValueError("Frozen confirmatory checkpoints changed")
    else:
        write_json(freeze_path, {"model_sha256": frozen,
                   "utc_time": pd.Timestamp.now(tz="UTC").isoformat(), "purpose": "evaluation only; no retraining, tuning or model selection"})
    records, provenance = [], {}
    for role in ("source", "target"):
        site = cfg[role]
        path = cache / f"{site}.json"
        params = {**CLIMATE_SITES[site], "start_date": "2024-12-25", "end_date": "2025-12-31",
                  "hourly": ",".join(cfg["columns"]), "timezone": "Asia/Kolkata"}
        if not path.exists():
            response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=55)
            response.raise_for_status()
            path.write_text(response.text, encoding="utf-8")
        js = json.loads(path.read_text(encoding="utf-8"))
        original = json.loads((RAW_DIR / "climate" / f"{site}.json").read_text(encoding="cp1252"))
        if js["hourly_units"] != original["hourly_units"]:
            raise ValueError("New weather units differ from training data")
        frame = pd.DataFrame(js["hourly"]).rename(columns={"time": "date"})
        frame["date"] = pd.to_datetime(frame["date"])
        values = frame[cfg["columns"]].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or not frame.date.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError("Confirmatory data must be finite and hourly; no interpolation allowed")
        frame.to_csv(cache / f"{site}.csv", index=False)
        scaler = Scaler(**meta[f"{role}_scaler"])
        x, y = make_windows([scaler.transform(values)], cfg["L"], cfg["H"])
        if frame.date.iloc[cfg["L"]] != pd.Timestamp("2025-01-01"):
            raise ValueError("Unexpected confirmatory forecast start")
        dom = {"test": (x, y), "scaler": scaler}
        denom = fixed_scale("climate", role, scaler)
        old = pd.read_csv(PROC_DIR / "climate" / f"{site}.csv", parse_dates=["date"])
        overlap = old.merge(frame, on="date", suffixes=("_old", "_new"))
        comparison = {name: float(np.sqrt(np.mean((overlap[f"{name}_old"] - overlap[f"{name}_new"]) ** 2))) for name in cfg["columns"]}
        provenance[role] = {"request": params, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "windows": len(x), "first_forecast_time": str(frame.date.iloc[cfg["L"]]),
                            "last_forecast_origin": str(frame.date.iloc[-cfg["H"]]),
                            "overlap_2024_RMSE_in_original_units": comparison,
                            "metadata": {k: v for k, v in js.items() if k != "hourly"},
                            "provider_note": "Original and new requests use API default best-match, not a pinned ERA5-only model."}
        checkpoint_name = "source_model" if role == "source" else "adapted_model"
        for seed in (0, 1, 2):
            strict = StrictEnsemble(meta, meta["model_spec"]["neural_spec"])
            strict.load_state_dict(torch.load(folder / f"seed_{seed}" / f"{checkpoint_name}.pt", weights_only=True))
            initial_meta = json.loads((MODEL_DIR / "improved" / "climate" / "meta.json").read_text())
            initial = make_forecaster(cfg, initial_meta["model_spec"])
            initial.load_state_dict(torch.load(MODEL_DIR / "improved" / "climate" / f"seed_{seed}" / f"{checkpoint_name}.pt", weights_only=True))
            legacy = make_forecaster(cfg)
            legacy.load_state_dict(torch.load(MODEL_DIR / "climate" / f"{checkpoint_name}.pt", weights_only=True))
            models = {"strict": strict, "previous_patch_mlp": initial, "legacy_seed0": legacy,
                      "seasonal_naive": seasonal_naive(x.numpy(), cfg["H"], cfg["season"]),
                      "persistence": seasonal_naive(x.numpy(), cfg["H"], 1),
                      "input_mean": np.repeat(x.numpy().mean(1, keepdims=True), cfg["H"], axis=1)}
            for method, model in models.items():
                scores, _ = evaluate("climate", cfg, model, dom, denom)
                records.append({"dataset": "climate", "period": "2025", "domain": role,
                                "seed": seed, "method": method, **scores})
        print(f"2025 {role}: evaluation complete", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "metrics_all.csv", index=False)
    write_json(out / "provenance.json", provenance)
    print(frame.groupby(["domain", "method"])[["MSE", "key_RMSE"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
