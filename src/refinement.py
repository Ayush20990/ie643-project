"""Full-context observed-history adaptation with channel-specific anchored ridge.

python -m src.refinement --datasets climate energy finance --seeds 0 1 2

Target train/validation future tensors are never supplied to fitting or selection.
Full L->H examples are reconstructed INSIDE the union of observed input windows.
This uses past observed outcomes as self-supervision, and is explicitly different
from the previous padded-prefix proxy. The old test period is a reused development
benchmark, not a fresh confirmatory holdout.
"""
import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, PROC_DIR, RESULTS_DIR
from .data import load_domain, make_windows, prepare, save_meta, series_list
from .improve import evaluate, fixed_scale, write_json
from .metrics import seasonal_naive
from .model import make_forecaster
from .refined_models import (ChannelRidge, NORMALIZATIONS, RefinedForecaster,
                             convex_weights, ridge_path, sufficient_statistics)
from .train import predict

PENALTIES = (1e-5, 1e-4, .001, .01, .1, 1.)
ANCHORS = (.0001, .001, .01, .1, 1., 10.)
REPLAY = (0., .25, 1.)


def reconstruct_inputs(inputs, counts, stride):
    """Reconstruct disjoint series from overlapping X only; never accept Y.

    Every returned value occurs in an actual supplied input window. Checking the
    overlap also rejects wrong series boundaries and noncontiguous window data.
    """
    if sum(counts) != len(inputs):
        raise ValueError("Input-window series counts do not match")
    arrays, start = [], 0
    for count in counts:
        group = inputs[start:start + count]
        if count < 1 or stride > group.shape[1]:
            raise ValueError("Cannot stitch empty or nonoverlapping windows")
        if count > 1 and not torch.equal(group[:-1, stride:], group[1:, :-stride]):
            raise ValueError("Windows are not consecutive observations of one series")
        arrays.append(torch.cat((group[0], group[1:, -stride:].reshape(-1, group.shape[2])), 0).numpy())
        start += count
    return arrays


def observed_examples(ds, inputs, split):
    cfg = DATASETS[ds]
    # Only lengths are read here. All fitted values come from the input tensor.
    arrays = series_list(load_domain(ds, cfg["target"]), cfg["columns"])
    stride = cfg["train_stride"] if split == "train" else 1
    lengths = []
    for a in arrays:
        cut = int(len(a) * cfg["split"][0])
        end = int(len(a) * sum(cfg["split"][:2]))
        lengths.append(cut if split == "train" else end - cut + cfg["L"])
    counts = [(n - cfg["L"] - cfg["H"]) // stride + 1 for n in lengths]
    observed = reconstruct_inputs(inputs, counts, stride)
    result = make_windows(observed, cfg["L"], cfg["H"], stride)
    audit = {"original_input_windows": len(inputs), "reconstructed_forecasts": len(result[0]),
             "observed_rows_per_series": [len(a) for a in observed],
             "last_H_future_rows_excluded": True, "target_Y_tensor_accessed": False}
    return result, audit


def per_channel_error(model, x, y):
    return np.mean((predict(model, x, bs=512) - y.numpy()) ** 2, axis=(0, 1))


def fit_source(cfg, source, out):
    C = len(cfg["columns"])
    best = np.full(C, np.inf)
    model = ChannelRidge(cfg["L"], cfg["H"], C)
    selection = [None] * C
    all_rows, statistics = [], {}
    for mode_index, mode in enumerate(NORMALIZATIONS):
        print(f"  source normal equations: {mode}", flush=True)
        g, r = sufficient_statistics(*source["train"], mode)
        statistics[mode] = (g, r)
        candidate = ChannelRidge(cfg["L"], cfg["H"], C)
        candidate.mode.fill_(mode_index)
        for penalty, weight in ridge_path(g, r, PENALTIES):
            candidate.weight.copy_(weight)
            error = per_channel_error(candidate, *source["val"])
            for c, name in enumerate(cfg["columns"]):
                row = {"column": name, "normalization": mode, "penalty": penalty, "validation_MSE": float(error[c])}
                all_rows.append(row)
                if error[c] < best[c]:
                    best[c] = error[c]
                    model.weight[c].copy_(weight[c])
                    model.mode[c] = mode_index
                    selection[c] = row
    pd.DataFrame(all_rows).to_csv(out / "source_search.csv", index=False)
    return model, selection, statistics


def fit_adapted(cfg, source_model, source_statistics, history_train, history_val, out):
    """L = target-history MSE + replay*source MSE + anchor*||W-W_source||².

    Fit exactly using normal equations. Modes come from source validation.
    Regularization/replay strengths are selected per channel on held-out observed
    history, including the unadapted source as an explicit fallback.
    """
    result = copy.deepcopy(source_model)
    best = per_channel_error(source_model, *history_val)
    selection = [{"column": name, "recipe": "source_fallback", "validation_MSE": float(best[c])}
                 for c, name in enumerate(cfg["columns"])]
    rows = list(selection)
    for mode_index, mode in enumerate(NORMALIZATIONS):
        channels = source_model.mode.numpy() == mode_index
        if not channels.any():
            continue
        print(f"  adaptation normal equations: {mode}", flush=True)
        gt, rt = sufficient_statistics(*history_train, mode)
        gs, rs = source_statistics[mode]
        candidate = copy.deepcopy(source_model)
        for replay in REPLAY:
            g, r = gt + replay * gs, rt + replay * rs
            for anchor, weight in ridge_path(g, r, ANCHORS, source_model.weight):
                candidate.weight.copy_(weight)
                error = per_channel_error(candidate, *history_val)
                for c, name in enumerate(cfg["columns"]):
                    if not channels[c]:
                        continue
                    row = {"column": name, "recipe": "anchored_history", "normalization": mode,
                           "source_replay": replay, "anchor": anchor, "validation_MSE": float(error[c])}
                    rows.append(row)
                    if error[c] < best[c]:
                        best[c] = error[c]
                        result.weight[c].copy_(weight[c])
                        selection[c] = row
    pd.DataFrame(rows).to_csv(out / "adaptation_search.csv", index=False)
    return result, selection


def calibrate(model, inputs, truth):
    parts = [predict(model.linear, inputs, bs=512), predict(model.neural, inputs, bs=512),
             np.repeat(inputs.numpy().mean(1, keepdims=True), model.H, axis=1),
             np.zeros_like(truth.numpy())]
    mix = convex_weights(parts, truth.numpy())
    model.mix.copy_(torch.from_numpy(mix))
    return {"weights_linear_neural_inputmean_trainingmean": mix.tolist(),
            "per_channel_validation_MSE": per_channel_error(model, inputs, truth).tolist()}


def run_dataset(ds, args):
    started = time.time()
    data = prepare(ds)
    cfg, src, tgt = data["cfg"], data["source"], data["target"]
    out, dest = RESULTS_DIR / "refined" / ds, MODEL_DIR / "refined" / ds
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": ds, "args": vars(args), "torch": torch.__version__, "python": platform.python_version(),
                "source_penalties": PENALTIES, "anchors": ANCHORS, "replay": REPLAY,
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((PROC_DIR / ds).glob("*.csv"))},
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__), Path(__file__).with_name("refined_models.py"))},
                "protocol": "source labels + forecasting entirely within observed target input history; no target future tensors in fitting/selection",
                "evaluation": "reused development test, previously observed in architecture experiments"}
    # JSON roundtrip normalizes tuples for consistent resume checks.
    manifest = json.loads(json.dumps(manifest))
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a distinct experiment directory")
    write_json(out / "manifest.json", manifest)
    history_train, audit_train = observed_examples(ds, tgt["train"][0], "train")
    history_val, audit_val = observed_examples(ds, tgt["val"][0], "val")
    write_json(out / "history_audit.json", {"train": audit_train, "validation": audit_val})
    print(f"{ds}: source fit; target history pairs train={len(history_train[0])}, val={len(history_val[0])}", flush=True)
    source, source_selection, source_statistics = fit_source(cfg, src, out)
    adapted, adapted_selection = fit_adapted(cfg, source, source_statistics, history_train, history_val, out)
    torch.save(source.state_dict(), dest / "source_linear.pt")
    torch.save(adapted.state_dict(), dest / "adapted_linear.pt")
    write_json(out / "linear_selection.json", {"source": source_selection, "adapted": adapted_selection})
    # Freeze ALL seeds and choices before any test forecast is computed.
    frozen = {}
    for seed in args.seeds:
        old = MODEL_DIR / "improved" / ds / f"seed_{seed}"
        selection = json.loads((RESULTS_DIR / "improved" / ds / f"seed_{seed}_selection.json").read_text())
        pair, choices = {}, {}
        for name, linear, checkpoint, validation in (
                ("source", source, "source_model.pt", src["val"]),
                ("adapted", adapted, "adapted_model.pt", history_val)):
            model = RefinedForecaster(cfg, selection["model_spec"])
            model.linear.load_state_dict(linear.state_dict())
            model.neural.load_state_dict(torch.load(old / checkpoint, weights_only=True))
            choices[name] = calibrate(model, *validation)
            pair[name] = model.eval()
        seed_dir = dest / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        for name, model in pair.items():
            torch.save(model.state_dict(), seed_dir / f"{name}_model.pt")
        frozen[seed] = pair
        choices.update(seed=seed, model_spec={"architecture": "refined", "neural_spec": selection["model_spec"]},
                       selection_used_test_futures=False, benchmark="reused development period")
        write_json(out / f"seed_{seed}_selection.json", choices)
        if seed == args.seeds[0]:
            for name, model in pair.items():
                torch.save(model.state_dict(), dest / f"{name}_model.pt")
            save_meta(ds, data, dest / "meta.json")
            meta = json.loads((dest / "meta.json").read_text())
            meta.update(model_spec=choices["model_spec"], training_seed=seed,
                        adaptation="anchored_full_history", experiment="refined")
            write_json(dest / "meta.json", meta)
            write_json(dest / "selection.json", choices)
    write_json(out / "selection_frozen.json", {"seeds": args.seeds, "test_evaluated_at_freeze": False,
               "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    print(f"{ds}: choices frozen; evaluating development test", flush=True)
    records, examples = [], {}
    for seed, pair in frozen.items():
        for role in ("source", "target"):
            dom = data[role]
            x, y = (v.numpy() for v in dom["test"])
            denom = fixed_scale(ds, role, dom["scaler"])
            models = {"refined_source": pair["source"], "refined_adapted": pair["adapted"],
                      "channel_linear_source": source, "channel_linear_adapted": adapted,
                      "seasonal_naive": seasonal_naive(x, cfg["H"], cfg["season"]),
                      "persistence": seasonal_naive(x, cfg["H"], 1),
                      "input_mean": np.repeat(x.mean(1, keepdims=True), cfg["H"], axis=1),
                      "training_mean": np.zeros_like(y)}
            indices = np.linspace(0, len(x) - 1, min(256, len(x)), dtype=int)
            for method, model in models.items():
                scores, p = evaluate(ds, cfg, model, dom, denom)
                records.append({"dataset": ds, "seed": seed, "domain": role, "method": method, **scores})
                if seed == args.seeds[0]:
                    examples[f"{role}_{method}"] = p[indices]
            if seed == args.seeds[0]:
                examples[f"{role}_X"], examples[f"{role}_Y"] = x[indices], y[indices]
                examples[f"{role}_indices"] = indices
    np.savez_compressed(out / "forecast_examples.npz", **examples)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "metrics_all.csv", index=False)
    frame.groupby(["dataset", "domain", "method"])[["MSE", "MAE", "key_RMSE", "MASE_train"]].agg(["mean", "std"]).to_csv(out / "summary.csv")
    write_json(out / "timing.json", {"seconds": time.time() - started})
    print(frame[frame.domain == "target"].groupby("method")[["MSE", "key_RMSE"]].mean().to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for ds in args.datasets:
        run_dataset(ds, args)
    frames = [pd.read_csv(RESULTS_DIR / "refined" / ds / "metrics_all.csv") for ds in args.datasets]
    pd.concat(frames, ignore_index=True).to_csv(RESULTS_DIR / "refined" / "metrics_all.csv", index=False)


if __name__ == "__main__":
    main()
