"""Audit completed experiment artifacts and checkpoint selection provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, PROC_DIR, RESULTS_DIR
from .improve import CANDIDATES
from .inference import load_bundle, load_dlinear_bundle
from .model import make_forecaster


def main():
    torch.set_num_threads(2)
    report = {}
    for ds, cfg in DATASETS.items():
        folder = RESULTS_DIR / "improved" / ds
        if not (folder / "manifest.json").exists():
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        for name, expected in manifest["data_sha256"].items():
            actual = hashlib.sha256((PROC_DIR / ds / name).read_bytes()).hexdigest()
            assert actual == expected, f"{ds}: dataset changed: {name}"
        complete = []
        for seed in manifest["args"]["seeds"]:
            csv_path = folder / f"seed_{seed}.csv"
            if not csv_path.exists():
                continue
            metrics = pd.read_csv(csv_path)
            assert not metrics.duplicated(["domain", "method"]).any()
            assert np.isfinite(metrics.select_dtypes("number").to_numpy()).all()
            assert len(metrics) == 30, f"{ds} seed {seed}: incomplete comparison table"
            selection = json.loads((folder / f"seed_{seed}_selection.json").read_text())
            source_choice = min(selection["source_val_mse"], key=selection["source_val_mse"].get)
            assert source_choice == selection["architecture"]
            for name, value in selection["source_val_mse"].items():
                history = json.loads((folder / f"seed_{seed}_{name}.json").read_text())
                assert np.isclose(value, min(h["val_mse"] for h in history))
            recipe = min(selection["adaptation_proxy"], key=selection["adaptation_proxy"].get)
            assert recipe == selection["adaptation"]
            weights = MODEL_DIR / "improved" / ds / f"seed_{seed}"
            for name in ("replay_inner", "replay_inner_mmd"):
                history = json.loads((folder / f"seed_{seed}_{name}.json").read_text())
                eligible = [h for h in history if h["eligible"]]
                chosen = min(eligible, key=lambda h: h["selection_score"])
                assert chosen["epoch"] == selection["epochs_selected"][name]
                assert np.isclose(chosen["selection_score"], selection["adaptation_proxy"][name])
                assert all("tgt_test_mse" not in h for h in history)
            selected = torch.load(weights / "adapted_model.pt", weights_only=True)
            reference = torch.load(weights / ("source_model.pt" if recipe == "source_only" else f"{recipe}.pt"), weights_only=True)
            assert all(torch.equal(selected[k], reference[k]) for k in selected)
            for name, spec in CANDIDATES.items():
                model = make_forecaster(cfg, spec)
                model.load_state_dict(torch.load(weights / f"{name}.pt", weights_only=True))
                with torch.no_grad():
                    p = model.eval()(torch.zeros(2, cfg["L"], len(cfg["columns"])))
                assert p.shape == (2, cfg["H"], len(cfg["columns"])) and torch.isfinite(p).all()
            complete.append(seed)
        if complete:
            meta, models = load_bundle(ds, True)
            assert meta["training_seed"] == manifest["args"]["seeds"][0]
            report[ds] = {"complete_seeds": complete, "expected_seeds": manifest["args"]["seeds"],
                          "architecture": meta["model_spec"], "adaptation": meta["adaptation"]}
    if not report:
        raise RuntimeError("No completed experiment artifacts")
    transfer_report = {}
    for ds in DATASETS:
        folder = RESULTS_DIR / "transfer_selection" / ds
        if not (folder / "manifest.json").exists():
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        actual_code = hashlib.sha256(Path(__file__).with_name("transfer_selection.py").read_bytes()).hexdigest()
        assert actual_code == manifest["code_sha256"], "Transfer-selection code differs from recorded run"
        complete = []
        for seed in manifest["seeds"]:
            if not (folder / f"seed_{seed}.csv").exists():
                continue
            metrics = pd.read_csv(folder / f"seed_{seed}.csv")
            assert len(metrics) == 18 and not metrics.duplicated(["domain", "method"]).any()
            assert np.isfinite(metrics.select_dtypes("number").to_numpy()).all()
            choice = json.loads((folder / f"seed_{seed}_selection.json").read_text())
            eligible = [n for n, v in choice["scores"].items() if v["eligible"]]
            expected = min(eligible, key=lambda n: choice["scores"][n]["proxy"]) if eligible else "legacy_source"
            assert expected == choice["chosen"]
            assert choice["future_target_values_used_for_selection"] is False
            for score in choice["scores"].values():
                proxy = .5 * score["source_val_mse"] / choice["source_reference_mse"] + .5 * score["target_input_mse"] / choice["input_reference_mse"]
                assert np.isclose(proxy, score["proxy"])
                assert score["eligible"] == (score["source_val_mse"] <= 1.1 * choice["source_reference_mse"])
            primary = MODEL_DIR / "improved" / ds / f"seed_{seed}"
            dest = MODEL_DIR / "transfer_selection" / ds / f"seed_{seed}"
            name = choice["chosen"]
            if name == "legacy_source":
                reference = MODEL_DIR / ds / "source_model.pt"
            elif name == "main_source":
                reference = primary / "source_model.pt"
            elif name == "dlinear_source":
                reference = primary / "dlinear.pt"
            else:
                reference = (primary if name.startswith("main_") else dest) / f"{choice['adaptation']}.pt"
            selected = torch.load(dest / "adapted_model.pt", weights_only=True)
            reference_state = torch.load(reference, weights_only=True)
            assert all(torch.equal(selected[k], reference_state[k]) for k in selected)
            complete.append(seed)
        if complete:
            meta, models = load_bundle(ds, True, True)
            _, dlinear_models = load_dlinear_bundle(ds)
            for model in [*models.values(), *dlinear_models.values()]:
                with torch.no_grad():
                    prediction = model(torch.zeros(2, meta["L"], len(meta["columns"])))
                assert prediction.shape == (2, meta["H"], len(meta["columns"])) and torch.isfinite(prediction).all()
            transfer_report[ds] = {"complete_seeds": complete, "expected_seeds": manifest["seeds"],
                                   "architecture": meta["model_spec"], "adaptation": meta["adaptation"]}
    report["transfer_selection"] = transfer_report
    output = RESULTS_DIR / "improved" / "artifact_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
