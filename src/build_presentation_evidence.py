"""Export saved numerical evidence for editable intensive-assessment charts.

No model fitting or example ranking: figures use the first saved forecast origin
and the first 96 rolling origins, both deployment seed 0. Tables use all seeds.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, RESULTS_DIR, MODEL_DIR


def main():
    sources = set()

    def read_csv(path):
        sources.add(path)
        return pd.read_csv(path)

    output = {
        "semantics": {
            "tables": "Three-seed target-domain means, unless explicitly seed 0",
            "plots": "Deployment seed 0; fixed first origin / first 96 rolling origins; no performance filtering",
            "H24": "One origin predicts 24 future hours, without new observations",
            "H1": "One future hour per origin, with refreshed observed history",
        },
        "datasets": {},
    }
    hourly = read_csv(RESULTS_DIR / "forecast_horizons/next_hour_headline.csv")
    daily = read_csv(RESULTS_DIR / "forecast_horizons/twenty_four_hour_headline.csv")
    for ds in ("climate", "energy"):
        record = {"hourly": hourly.set_index("dataset").loc[ds].to_dict(),
                  "daily": daily.set_index("dataset").loc[ds].to_dict()}
        for family, filename, role in (("event_boost", "forecast_examples.npz", "H24"),
                                       ("peak_next_hour", "rolling_forecasts.npz", "H1")):
            archive = RESULTS_DIR / family / ds / filename
            meta_path = MODEL_DIR / family / ds / "meta.json"
            sources.update((archive, meta_path))
            a = np.load(archive)
            meta = json.loads(meta_path.read_text())
            key = meta["columns"].index(meta["key"])
            sd, mean = meta["target_scaler"]["std"][key], meta["target_scaler"]["mean"][key]
            def units(v):
                values = np.asarray(v, dtype=float) * sd + mean
                assert np.isfinite(values).all()
                return values.tolist()
            if role == "H24":
                earlier_path = RESULTS_DIR / ("strict_context" if ds == "climate" else "strict_refined") / ds / "forecast_examples.npz"
                sources.add(earlier_path)
                earlier = np.load(earlier_path)
                assert a["target_indices"][0] == earlier["target_indices"][0] == 0
                assert np.allclose(a["target_Y"][0], earlier["target_Y"][0])
                record[role] = {
                    "origin": int(a["target_indices"][0]),
                    "hours": list(range(1, 25)),
                    "actual": units(a["target_Y"][0, :, key]),
                    "retained": units(a["target_retained_backbone"][0, :, key]),
                    "adapted": units(a["target_adapted_model"][0, :, key]),
                    "earlier": units(earlier["target_strict_adapted"][0, :, key]),
                }
            else:
                record[role] = {
                    "origin": 0,
                    "hours": list(range(1, 97)),
                    "actual": units(a["target_truth"][:96, 0, key]),
                    "source": units(a["target_source_model"][:96, 0, key]),
                    "adapted": units(a["target_adapted_model"][:96, 0, key]),
                    "persistence": units(a["target_X"][:96, -1, key]),
                }
        peak = read_csv(RESULTS_DIR / "forecast_horizons" / f"{ds}_next_hour_peak_comparison.csv")
        record["events"] = peak.to_dict(orient="records")
        strict = read_csv(RESULTS_DIR / ("strict_context" if ds == "climate" else "strict_refined") / ds / "metrics_all.csv")
        method = "strict_adapted"
        chosen = strict.query("domain == 'target' and method == @method")
        naive = strict.query("domain == 'target' and method == 'seasonal_naive'")
        record["earlier_hybrid"] = {"MSE": chosen.MSE.mean(), "key_RMSE": chosen.key_RMSE.mean()}
        record["seasonal_naive"] = {"MSE": naive.MSE.mean(), "key_RMSE": naive.key_RMSE.mean()}
        output["datasets"][ds] = record
    f = read_csv(RESULTS_DIR / "strict_refined/finance/metrics_all.csv")
    output["finance"] = {method: f.query("domain == 'target' and method == @method")[["MSE", "key_RMSE"]].mean().to_dict()
                         for method in ("strict_source", "strict_adapted", "input_mean")}
    output["sources"] = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(sources)}
    out = ROOT / "presentation/intensive/evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
