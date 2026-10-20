"""Audit retained-LiteTCN experiments and measure deployment footprint/latency."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, PROC_DIR, RESULTS_DIR
from .improve import write_json
from .inference import load_bundle, load_peak_bundle, load_residual_bundle
from .peak_data import prepare_peak
from .peak_train import CANDIDATES
from .residual_tcn import ResidualLiteTCN
from .train import predict


def main():
    torch.set_num_threads(2)
    audits, timing = {}, []
    for ds in ("climate", "energy"):
        cfg = DATASETS[ds]
        out = RESULTS_DIR/"peak_litetcn"/ds
        manifest = json.loads((out/"manifest.json").read_text())
        for name, checksum in manifest["code_sha256"].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == checksum, name
        for name, checksum in manifest["data_sha256"].items():
            assert hashlib.sha256((PROC_DIR/ds/name).read_bytes()).hexdigest() == checksum
        choice = json.loads((out/"source_architecture_frozen.json").read_text())
        assert choice["chosen"] == min(choice["source_scores"], key=choice["source_scores"].get)
        for name in CANDIDATES:
            screen = json.loads((out/f"screen_0_{name}.json").read_text())
            assert screen["selected_round"] == min(screen["history"], key=lambda r: r["score"])["round"]
        for seed in (0, 1, 2):
            selection = json.loads((out/f"seed_{seed}_selection.json").read_text())
            assert selection["target_future_labels_used_for_training_or_selection"] is False
            assert selection["adaptation"] == min(selection["proxy"], key=selection["proxy"].get)
            for name in ("input_mse", "input_shape"):
                history = json.loads((out/f"seed_{seed}_{name}.json").read_text())
                best = min((r for r in history if r["eligible"]), key=lambda r: r["proxy"])
                assert best["round"] == selection["selected_rounds"][name]
                assert np.isclose(best["proxy"], selection["proxy"][name])
                for row in history:
                    assert row["eligible"] == (row["source"]["MSE"] <= 1.1*history[0]["source"]["MSE"])
            directory = MODEL_DIR/"peak_litetcn"/ds/f"seed_{seed}"
            selected = torch.load(directory/"adapted_model.pt", weights_only=True)
            reference = torch.load(directory/("source_model.pt" if selection["adaptation"] == "source_only" else selection["adaptation"]+".pt"), weights_only=True)
            assert all(torch.equal(selected[k], reference[k]) for k in selected)
        for family, expected_rows in (("peak_litetcn", 30), ("peak_calibrated", 24), ("residual_litetcn", 18)):
            folder = RESULTS_DIR/family/ds
            frame = pd.read_csv(folder/"metrics_all.csv")
            assert len(frame) == expected_rows and not frame.duplicated(["domain", "seed", "method"]).any()
            assert np.isfinite(frame.select_dtypes("number")).all().all()
            assert json.loads((folder/"selection_frozen.json").read_text())["test_evaluated_at_freeze"] is False
            if family != "peak_litetcn":
                record = json.loads((folder/"manifest.json").read_text())
                for name, checksum in record["code_sha256"].items():
                    assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == checksum
        for seed in (0, 1, 2):
            folder = RESULTS_DIR/"peak_calibrated"/ds
            selection = json.loads((folder/f"seed_{seed}_selection.json").read_text())
            for role in ("source", "adapted"):
                search = pd.read_csv(folder/f"seed_{seed}_{role}_search.csv")
                feasible = search[search.eligible]
                winner = feasible.sort_values("source_score" if role == "source" else "proxy", kind="stable").iloc[0]
                config = selection["calibration_source" if role == "source" else "calibration_target"]
                assert all(config[k] == winner[k] for k in config)
            folder = MODEL_DIR/"residual_litetcn"/ds/f"seed_{seed}"
            source = torch.load(folder/"source_model.pt", weights_only=True)
            warm = torch.load(MODEL_DIR/"improved"/ds/f"seed_{seed}"/"tcn3.pt", weights_only=True)
            assert all(torch.equal(v, warm[k]) for k, v in source.items() if k.startswith(("enc.", "proj.")))
            selection = json.loads((RESULTS_DIR/"residual_litetcn"/ds/f"seed_{seed}_selection.json").read_text())
            assert selection["target_forecasting_labels_used"] is False
            adapted = ResidualLiteTCN(cfg)
            adapted.load_state_dict(torch.load(folder/"adapted_model.pt", weights_only=True))
            weights = adapted.head_weights().detach()
            for channel, choice in enumerate(selection["adapted_head"]):
                assert choice["proxy"] <= 1.00001
                if choice["method"] == "zero_residual":
                    assert not bool(weights[channel].any())
                if "eligible" in choice:
                    assert choice["eligible"]
        data = prepare_peak(ds)
        _, legacy = load_bundle(ds)
        deployed = {"original_LiteTCN": legacy, "LiteTCN_v3": load_peak_bundle(ds)[1],
                    "LiteTCN_v3_local": load_peak_bundle(ds, True)[1], "LiteTCN_R": load_residual_bundle(ds)[1]}
        for family, pair in deployed.items():
            for role in ("source", "target"):
                model = pair["source_model" if role == "source" else "adapted_model"]
                inputs = data[role]["test_packed"][0] if getattr(model, "requires_calendar", False) else data[role]["test"][0]
                if family != "original_LiteTCN":
                    group, method = {"LiteTCN_v3": ("peak_litetcn", "source_only" if role == "source" else "selected_adapted"),
                        "LiteTCN_v3_local": ("peak_calibrated", "local_source" if role == "source" else "local_adapted"),
                        "LiteTCN_R": ("residual_litetcn", "source_residual_tcn" if role == "source" else "adapted_residual_tcn")}[family]
                    table = pd.read_csv(RESULTS_DIR/group/ds/"metrics_all.csv")
                    expected = table[(table.seed == 0)&(table.domain == role)&(table.method == method)].MSE.iloc[0]
                    measured = float(np.mean((predict(model, inputs, bs=256)-data[role]["test"][1].numpy())**2))
                    assert np.isclose(measured, expected, rtol=1e-5), (family, ds, role)
            model = pair["adapted_model"]
            inputs = data["target"]["test_packed"][0][:1] if getattr(model, "requires_calendar", False) else data["target"]["test"][0][:1]
            torch.set_num_threads(1)
            elapsed = []
            with torch.no_grad():
                for _ in range(5): model(inputs)
                for _ in range(30):
                    start = time.perf_counter()
                    model(inputs)
                    elapsed.append((time.perf_counter()-start)*1000)
            parameters = sum(p.numel() for p in model.parameters())
            auxiliary = sum(p.numel() for n, p in model.named_parameters() if n.startswith("recon."))
            timing.append({"dataset": ds, "model": family, "parameters_including_auxiliary": parameters,
                           "forecast_parameters": parameters-auxiliary,
                           "trainable_parameters_in_saved_architecture": sum(p.numel() for p in model.parameters() if p.requires_grad),
                           "CPU_threads": 1, "median_single_window_ms": float(np.median(elapsed))})
            torch.set_num_threads(2)
        audits[ds] = {"seeds": [0, 1, 2], "code_and_data_hashes": True, "selection_recomputed": True,
                      "original_encoder_retained_in_LiteTCN_R": True, "deployment_test_MSE_recomputed": True}
        print(f"audited {ds}", flush=True)
    fresh = RESULTS_DIR/"peak_litetcn"/"climate_2026"
    frozen = json.loads((fresh/"frozen_before_download.json").read_text())
    for name, checksum in frozen["model_sha256"].items():
        assert hashlib.sha256((MODEL_DIR/name).read_bytes()).hexdigest() == checksum
    audits["new_2026_climate_weights_remain_frozen"] = True
    write_json(RESULTS_DIR/"peak_litetcn"/"artifact_audit.json", audits)
    pd.DataFrame(timing).to_csv(RESULTS_DIR/"peak_litetcn"/"efficiency.csv", index=False)
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
