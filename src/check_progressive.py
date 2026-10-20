"""Recompute selection and checkpoint/data invariants for gradual unfreezing."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, PROC_DIR, RESULTS_DIR
from .data import prepare
from .improve import write_json
from .inference import load_progressive_bundle
from .progressive_train import FAMILY, MODES, quality, sha


def main():
    torch.set_num_threads(2)
    audits, efficiency = {}, []
    for ds in ("climate", "energy"):
        out, dest = RESULTS_DIR/FAMILY/ds, MODEL_DIR/FAMILY/ds
        manifest = json.loads((out/"manifest.json").read_text())
        for name, checksum in manifest["code_sha256"].items():
            assert sha(Path(__file__).with_name(name)) == checksum, name
        for name, checksum in manifest["data_sha256"].items():
            assert sha(PROC_DIR/ds/name) == checksum
        for path, checksum in manifest["initial_checkpoint_sha256"].items():
            assert sha(MODEL_DIR/path) == checksum
        freeze = json.loads((out/"selection_frozen.json").read_text())
        assert freeze["test_evaluated_at_freeze"] is False
        for path, checksum in freeze["checkpoint_sha256"].items():
            assert sha(dest/path) == checksum
        frame = pd.read_csv(out/"metrics_all.csv")
        assert len(frame) == 24 and not frame.duplicated(["seed","domain","method"]).any()
        assert np.isfinite(frame.select_dtypes("number")).all().all()
        assert (out/"metrics_all.csv").stat().st_mtime >= pd.Timestamp(freeze["timestamp_utc"]).timestamp()
        seed_records = []
        for seed in (0,1,2):
            selected = json.loads((out/f"seed_{seed}_selection.json").read_text())
            assert selected["target_future_labels_used_for_training_or_selection"] is False
            scores = {}
            for mode in MODES:
                record = json.loads((out/f"seed_{seed}_{mode}.json").read_text())
                best = min((r for r in record["history"] if r["eligible"]), key=lambda r:r["score"])
                assert record["selected_round"] == best["round"]
                assert np.isclose(record["score"], best["score"])
                for r in record["history"]:
                    assert r["eligible"] == (r["MSE"] <= 1.03*record["history"][0]["MSE"])
                scores[mode] = record["score"]
            assert selected["source_control"] == min(scores, key=scores.get)
            seed_dir = dest/f"seed_{seed}"
            source = torch.load(seed_dir/"source_model.pt", weights_only=True)
            candidate = torch.load(seed_dir/f"{selected['source_control']}.pt", weights_only=True)
            assert all(torch.equal(v,candidate[n]) for n,v in source.items())
            history = json.loads((out/f"seed_{seed}_adaptation.json").read_text())["history"]
            best = min((r for r in history if r["eligible"]), key=lambda r:r["proxy"])
            assert selected["adaptation_round"] == best["round"]
            for r in history:
                assert r["eligible"] == (r["source"]["MSE"] <= 1.1*history[0]["source"]["MSE"])
                expected = .5*r["source"]["score"]/history[0]["source"]["score"]+.5*r["target_input"]["score"]/history[0]["target_input"]["score"]
                assert np.isclose(expected, r["proxy"])
            initial = torch.load(MODEL_DIR/"residual_litetcn"/ds/f"seed_{seed}"/"source_model.pt", weights_only=True)
            for name in (*MODES,"source_model","adapted_model"):
                weights = torch.load(seed_dir/f"{name}.pt", weights_only=True)
                assert all(torch.equal(v,initial[n]) for n,v in weights.items() if n.startswith(("linear.","revin.")))
            adapted = torch.load(seed_dir/"adapted_model.pt", weights_only=True)
            encoder_drift = sum(float((v-source[n]).square().sum()) for n,v in adapted.items() if n.startswith("enc."))**.5
            if selected["adaptation_round"] > 4:
                assert encoder_drift > 0
            if selected["adaptation_round"] == 0:
                assert all(torch.equal(v,source[n]) for n,v in adapted.items())
            seed_records.append({"seed":seed,"source_control":selected["source_control"],
                "adaptation_round":selected["adaptation_round"],"adapted_encoder_L2_change":encoder_drift})
        meta, models = load_progressive_bundle(ds)
        for role in ("source_model","adapted_model"):
            assert sha(dest/f"{role}.pt") == sha(dest/"seed_0"/f"{role}.pt")
        data = prepare(ds)
        key = DATASETS[ds]["columns"].index(DATASETS[ds]["key"])
        for role, model_name in (("source","source_model"),("target","adapted_model")):
            actual = quality(models[model_name], *data[role]["test"], key)["MSE"]
            expected = frame.query("seed==0 and domain==@role and method==@model_name").MSE.item()
            assert np.isclose(actual, expected, rtol=1e-5)
        model = models["adapted_model"]
        torch.set_num_threads(1)
        sample = data["target"]["test"][0][:1]
        with torch.no_grad():
            for _ in range(5): model(sample)
            start = time.perf_counter()
            for _ in range(30): model(sample)
        efficiency.append({"dataset":ds,"parameters":sum(p.numel() for p in model.parameters()),
            "batch_1_CPU_ms":(time.perf_counter()-start)/30*1000,"threads":1,"warmups":5,"repeats":30})
        torch.set_num_threads(2)
        audits[ds] = {"data_code_initialization_hashes":True,"selection_recomputed":True,
            "skip_and_RevIN_unchanged":True,"full_deployment_test_MSE_recomputed":True,"seeds":seed_records}
    fresh = RESULTS_DIR/FAMILY/"climate_oct2026"/"frozen_before_download.json"
    if fresh.exists():
        record = json.loads(fresh.read_text())
        assert all(sha(MODEL_DIR/p)==checksum for p,checksum in record["model_sha256"].items())
        audits["fresh_climate_models_unchanged"] = True
    write_json(RESULTS_DIR/FAMILY/"artifact_audit.json",audits)
    pd.DataFrame(efficiency).to_csv(RESULTS_DIR/FAMILY/"efficiency.csv",index=False)
    print(json.dumps(audits,indent=2))


if __name__ == "__main__":
    main()
