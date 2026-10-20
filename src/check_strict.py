"""Audit strict experiment data/code hashes, checkpoint calibration and efficiency."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, PROC_DIR, RESULTS_DIR
from .data import prepare
from .improve import write_json
from .inference import load_bundle, load_strict_bundle
from .strict_models import input_pretext
from .train import predict


def main():
    torch.set_num_threads(2)
    report, timing = {}, []
    for context, family in ((False, "strict_refined"), (True, "strict_context")):
        report[family] = {}
        for ds, cfg in DATASETS.items():
            out = RESULTS_DIR / family / ds
            manifest = json.loads((out / "manifest.json").read_text())
            for name, expected in manifest["code_sha256"].items():
                assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected, (ds, name)
            for name, expected in manifest["data_sha256"].items():
                assert hashlib.sha256((PROC_DIR / ds / name).read_bytes()).hexdigest() == expected, (ds, name)
            frame = pd.read_csv(out / "metrics_all.csv")
            assert len(frame) == 42 and not frame.duplicated(["seed", "domain", "method"]).any()
            assert np.isfinite(frame.select_dtypes("number")).all().all()
            freeze = json.loads((out / "selection_frozen.json").read_text())
            assert freeze["test_evaluated_at_freeze"] is False
            for seed in manifest["args"]["seeds"]:
                choice = json.loads((out / f"seed_{seed}_selection.json").read_text())
                assert choice["target_future_labels_used_for_training_or_selection"] is False
                assert choice["cross_window_target_label_reconstruction"] is False
                for row in choice["calibration"]:
                    assert row["source_val_MSE"] <= 1.1 * row["source_reference_MSE"] + 1e-8
                    assert row["selection_score"] <= 1.00001
                state = torch.load(MODEL_DIR / family / ds / f"seed_{seed}" / "adapted_model.pt", weights_only=True)
                np.testing.assert_allclose(state["mix"].numpy(), choice["adapted_weights"], atol=1e-7)
                np.testing.assert_allclose(state["mix"].sum(-1).numpy(), 1, atol=1e-6)
                assert bool((state["mix"] >= 0).all())
            meta, models = load_strict_bundle(ds, context)
            data = prepare(ds)
            # Recompute recorded retention and input-only pretext errors from deployment weights.
            selected = json.loads((out / "seed_0_selection.json").read_text())
            source = data["source"]
            ps = predict(models["adapted_model"], source["val"][0], bs=256)
            sv = ((ps - source["val"][1].numpy()) ** 2).mean((0, 1))
            px, py = input_pretext(data["target"]["val"][0], cfg["H"])
            pt = predict(models["adapted_model"], px, bs=256)
            tv = ((pt - py.numpy()) ** 2).mean((0, 1))
            np.testing.assert_allclose(sv, [r["source_val_MSE"] for r in selected["calibration"]], rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(tv, [r["target_input_pretext_MSE"] for r in selected["calibration"]], rtol=1e-4, atol=1e-6)
            for role in ("source", "target"):
                name = "source_model" if role == "source" else "adapted_model"
                method = "strict_source" if role == "source" else "strict_adapted"
                expected = frame[(frame.seed == 0) & (frame.domain == role) & (frame.method == method)].MSE.iloc[0]
                actual = float(((predict(models[name], data[role]["test"][0], bs=256) - data[role]["test"][1].numpy()) ** 2).mean())
                assert np.isclose(actual, expected, rtol=1e-5), (family, ds, role)
            _, legacy = load_bundle(ds)
            torch.set_num_threads(1)
            for name, model in ((family, models["adapted_model"]), ("legacy", legacy["adapted_model"])):
                x = data["target"]["test"][0][:1]
                elapsed = []
                with torch.no_grad():
                    for _ in range(5):
                        model(x)
                    for _ in range(30):
                        start = time.perf_counter()
                        model(x)
                        elapsed.append((time.perf_counter() - start) * 1000)
                timing.append({"dataset": ds, "family": name, "single_window_median_ms": float(np.median(elapsed)),
                               "stored_float_coefficients": sum(v.numel() for v in model.state_dict().values() if v.is_floating_point()),
                               "cpu_threads": 1, "warmups": 5, "repeats": 30})
            torch.set_num_threads(2)
            report[family][ds] = {"seeds": [0, 1, 2], "metrics_rows": len(frame), "hashes_verified": True,
                                  "source_guard_recomputed": True, "test_MSE_recomputed": True}
            print(f"audited {family}/{ds}", flush=True)
    fresh = RESULTS_DIR / "strict_refined" / "confirmatory_2025"
    frozen = json.loads((fresh / "freeze_before_download.json").read_text())
    for name, expected in frozen["model_sha256"].items():
        assert hashlib.sha256((MODEL_DIR / name).read_bytes()).hexdigest() == expected
    metrics = pd.read_csv(fresh / "metrics_all.csv")
    assert len(metrics) == 36 and np.isfinite(metrics.select_dtypes("number")).all().all()
    report["climate_2025_frozen_models_verified"] = True
    write_json(RESULTS_DIR / "strict_refined" / "artifact_audit.json", report)
    pd.DataFrame(timing).to_csv(RESULTS_DIR / "strict_refined" / "efficiency.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
