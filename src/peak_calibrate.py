"""Select input-only residual correction before evaluating development futures."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, RESULTS_DIR
from .diagnose_peaks import peak_metrics
from .improve import evaluate, fixed_scale, write_json
from .peak_calibration import CalibratedLiteTCN, correct_prediction
from .peak_data import prepare_peak
from .peak_train import make_peak, quality
from .train import predict

CONFIGS = [{"kind": "level", "strength": 0., "decay": 24.}] + [
    {"kind": kind, "strength": strength, "decay": decay}
    for kind in ("level", "innovation", "affine") for strength in (.25, .5, 1.) for decay in (6., 24.)]


def cached_predictions(base, inputs, prefix=False):
    inputs = inputs[:, :-12] if prefix else inputs
    if inputs.shape[1]-12 < base.context:
        raise ValueError("Residual-calibration validation needs two observed 12-hour blocks")
    prediction = torch.from_numpy(predict(base, inputs, bs=256))
    past = torch.from_numpy(predict(base, inputs[:, :-12], bs=256))[:, :12]
    observed = inputs[:, -12:, :base.C]
    return prediction, past, observed


def select_calibration(base, source_val, target_val_inputs, cfg):
    """No target Y parameter exists; nested target validation stays inside X."""
    source = cached_predictions(base, source_val[0])
    target = cached_predictions(base, target_val_inputs, prefix=True)
    target_observed = target_val_inputs[:, -12:, :base.C]
    key = cfg["columns"].index(cfg["key"])
    bs = quality(source[0].numpy(), source_val[1].numpy(), key)
    bt = quality(target[0][:, :12].numpy(), target_observed.numpy(), key)
    rows, best_source, best_target = [], 0, 0
    source_score, target_score = float("inf"), 1.
    for index, config in enumerate(CONFIGS):
        ps = correct_prediction(*source, **config)
        pt = correct_prediction(*target, **config)[:, :12]
        sv = quality(ps.numpy(), source_val[1].numpy(), key)
        tv = quality(pt.numpy(), target_observed.numpy(), key)
        proxy = .5*sv["score"]/bs["score"] + .5*tv["score"]/bt["score"]
        eligible = sv["MSE"] <= 1.1*bs["MSE"]
        rows.append({"configuration": index, **config, "source_MSE": sv["MSE"],
                     "source_score": sv["score"], "target_input_MSE": tv["MSE"],
                     "target_input_score": tv["score"], "proxy": proxy, "eligible": eligible})
        if eligible and sv["score"] < source_score:
            source_score, best_source = sv["score"], index
        if eligible and proxy < target_score:
            target_score, best_target = proxy, index
    return CONFIGS[best_source], CONFIGS[best_target], rows


def run_dataset(ds, seeds):
    data = prepare_peak(ds)
    cfg, src, tgt = data["cfg"], data["source"], data["target"]
    out, dest = RESULTS_DIR/"peak_calibrated"/ds, MODEL_DIR/"peak_calibrated"/ds
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    source_folder = MODEL_DIR/"peak_litetcn"/ds
    meta = json.loads((source_folder/"meta.json").read_text())
    manifest = {"dataset": ds, "seeds": seeds, "configurations": CONFIGS,
                "code_sha256": {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                for n in ("peak_calibrate.py", "peak_calibration.py")},
                "base_checkpoints": {str(p.relative_to(MODEL_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in source_folder.glob("seed_*/*_model.pt")},
                "target_protocol": "nested 12h input-only validation; no target forecasting labels; independent per-window correction",
                "evaluation": "exploratory reused development test, not a fresh holdout"}
    write_json(out/"manifest.json", manifest)
    frozen = {}
    for seed in seeds:
        print(f"{ds} s{seed}: select recent-residual correction using source val and target inputs", flush=True)
        models, configs = {}, {}
        for role in ("source", "adapted"):
            base = make_peak(cfg, meta["model_spec"])
            base.load_state_dict(torch.load(source_folder/f"seed_{seed}"/f"{role}_model.pt", weights_only=True))
            base.eval()
            source_config, target_config, rows = select_calibration(base, src["val_packed"], tgt["val_packed"][0], cfg)
            config = source_config if role == "source" else target_config
            models[f"raw_{role}"] = base
            models[f"local_{role}"] = CalibratedLiteTCN(copy.deepcopy(base), **config).eval()
            configs[role] = config
            pd.DataFrame(rows).to_csv(out/f"seed_{seed}_{role}_search.csv", index=False)
        seed_dir = dest/f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        for role in ("source", "adapted"):
            torch.save(models[f"local_{role}"].state_dict(), seed_dir/f"{role}_model.pt")
        selection = {"seed": seed, "calibration_source": configs["source"], "calibration_target": configs["adapted"],
                     "target_forecasting_labels_used": False, "calibration_inside_each_input_only": True}
        write_json(out/f"seed_{seed}_selection.json", selection)
        frozen[seed] = models
        if seed == seeds[0]:
            for role in ("source", "adapted"):
                torch.save(models[f"local_{role}"].state_dict(), dest/f"{role}_model.pt")
            deployed = {**meta, **selection, "experiment": "peak_calibrated"}
            write_json(dest/"meta.json", deployed)
            write_json(dest/"selection.json", selection)
    write_json(out/"selection_frozen.json", {"test_evaluated_at_freeze": False,
               "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    rows, peaks, examples = [], [], {}
    for seed, models in frozen.items():
        for role in ("source", "target"):
            dom = data[role]
            indices = np.linspace(0, len(dom["test"][0])-1, min(256, len(dom["test"][0])), dtype=int)
            key = cfg["columns"].index(cfg["key"])
            for name, model in models.items():
                p = predict(model, dom["test_packed"][0], bs=256)
                scores, _ = evaluate(ds, cfg, p, dom, fixed_scale(ds, role, dom["scaler"]))
                rows.append({"dataset": ds, "seed": seed, "domain": role, "method": name, **scores})
                sd, mu = dom["scaler"].std[key], dom["scaler"].mean[key]
                peaks.append({"dataset": ds, "seed": seed, "domain": role, "method": name,
                              **peak_metrics(dom["test"][1].numpy()[..., key]*sd+mu, p[..., key]*sd+mu)})
                if seed == seeds[0]:
                    examples[f"{role}_{name}"] = p[indices]
            if seed == seeds[0]:
                examples[f"{role}_X"], examples[f"{role}_Y"] = (a.numpy()[indices] for a in dom["test"])
                examples[f"{role}_indices"] = indices
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"metrics_all.csv", index=False)
    pd.DataFrame(peaks).to_csv(out/"peak_metrics.csv", index=False)
    np.savez_compressed(out/"forecast_examples.npz", **examples)
    print(frame.query("domain == 'target'").groupby("method")[["MSE", "key_RMSE"]].mean().to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("climate", "energy"), default=["climate", "energy"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    torch.set_num_threads(2)
    for ds in args.datasets:
        run_dataset(ds, args.seeds)


if __name__ == "__main__":
    main()
