"""Strict label-free context experiment: a padding-free linear pretext branch.

python -m src.strict_context --datasets climate energy finance --seeds 0 1 2

No original target forecasting labels are reconstructed from overlapping windows.
New channel ridge adapts using source replay, an L2 source anchor, and a real
L-H prefix inside each target input. Neural branch proxies retain their original padding. A
source-guarded per-channel convex ensemble uses source validation and target input
pretext scores. All choices freeze before reused development-test evaluation.
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
from scipy.optimize import minimize

from .config import DATASETS, MODEL_DIR, PROC_DIR, RESULTS_DIR
from .data import batches, prepare, save_meta
from .improve import evaluate, fixed_scale, write_json
from .metrics import seasonal_naive
from .refined_models import convex_weights, ridge_path, sufficient_statistics, NORMALIZATIONS
from .refinement import per_channel_error, ANCHORS, REPLAY
from .context_models import ContextEnsemble as StrictEnsemble, fit_context_source as fit_source, real_context_pretext
from .strict_models import BRANCHES, COMPONENTS, input_pretext
from .train import predict


def fit_input_only(source_model, source, source_stats, target_train_inputs, target_val_inputs, out):
    train = real_context_pretext(target_train_inputs, source_model.H)
    validation = real_context_pretext(target_val_inputs, source_model.H)
    base_src = per_channel_error(source_model, *source["val"])
    base_pretext = per_channel_error(source_model, *validation)
    best = np.ones(source_model.C)
    result = copy.deepcopy(source_model)
    choices = [{"column_index": c, "recipe": "source_fallback", "score": 1.} for c in range(source_model.C)]
    rows = []
    for mode_index, mode in enumerate(NORMALIZATIONS):
        channels = source_model.mode.numpy() == mode_index
        if not channels.any():
            continue
        print(f"  strict within-input adapter: {mode}", flush=True)
        gt, rt = sufficient_statistics(*train, mode)
        gs, rs = source_stats[mode]
        candidate = copy.deepcopy(source_model)
        for replay in REPLAY:
            for anchor, weight in ridge_path(gt + replay * gs, rt + replay * rs, ANCHORS, source_model.weight):
                candidate.weight.copy_(weight)
                sv = per_channel_error(candidate, *source["val"])
                tv = per_channel_error(candidate, *validation)
                score = .5 * sv / np.maximum(base_src, 1e-8) + .5 * tv / np.maximum(base_pretext, 1e-8)
                for c in np.flatnonzero(channels):
                    eligible = bool(sv[c] <= 1.1 * base_src[c])
                    row = {"column_index": int(c), "normalization": mode, "replay": replay, "anchor": anchor,
                           "source_val_MSE": float(sv[c]), "target_input_pretext_MSE": float(tv[c]),
                           "score": float(score[c]), "eligible": eligible}
                    rows.append(row)
                    if eligible and score[c] < best[c]:
                        best[c] = score[c]
                        result.weight[c].copy_(weight[c])
                        choices[c] = row
    pd.DataFrame(rows).to_csv(out / "input_adaptation_search.csv", index=False)
    return result, choices


@torch.no_grad()
def component_predictions(model, inputs):
    return torch.cat([model.components(x) for x in batches(inputs, bs=256, shuffle=False)]).numpy()


def source_calibration(model, source_val):
    p = component_predictions(model, source_val[0])
    indices = [0, 2, 4, 6, 7]
    w = convex_weights([p[..., i] for i in indices], source_val[1].numpy())
    model.mix.zero_()
    model.mix[:, indices] = torch.from_numpy(w)
    return p


def strict_calibration(model, source_components, source_truth, target_inputs):
    """Optimize normalized source + within-input pretext error, with source guard.

    The source mixture (adapters weighted zero) is feasible by construction.
    The API accepts target X only, never target Y or reconstructed future labels.
    """
    context, truth = input_pretext(target_inputs, model.H)
    target_components = component_predictions(model, context)
    source_weights = model.mix.numpy().copy()
    records = []
    for c in range(model.C):
        ps = source_components[..., c, :].reshape(-1, len(COMPONENTS)).astype(np.float64)
        pt = target_components[..., c, :].reshape(-1, len(COMPONENTS)).astype(np.float64)
        ys, yt = source_truth.numpy()[..., c].reshape(-1), truth.numpy()[..., c].reshape(-1)
        initial = source_weights[c].astype(np.float64)
        gs, rs, ss = ps.T @ ps / len(ys), ps.T @ ys / len(ys), float(np.mean(ys.astype(np.float64) ** 2))
        gt, rt, tt = pt.T @ pt / len(yt), pt.T @ yt / len(yt), float(np.mean(yt.astype(np.float64) ** 2))
        error_s = lambda w: float(w @ gs @ w - 2 * w @ rs + ss)
        error_t = lambda w: float(w @ gt @ w - 2 * w @ rt + tt)
        bs, bt = max(error_s(initial), 1e-8), max(error_t(initial), 1e-8)
        score = lambda w: .5 * error_s(w) / bs + .5 * error_t(w) / bt + .001 * np.sum((w - initial) ** 2)
        jac = lambda w: (gs @ w - rs) / bs + (gt @ w - rt) / bt + .002 * (w - initial)
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
                       {"type": "ineq", "fun": lambda w: 1.1 * bs - error_s(w), "jac": lambda w: -2 * (gs @ w - rs)}]
        fit = minimize(score, initial, jac=jac, bounds=[(0, 1)] * len(COMPONENTS), constraints=constraints,
                       method="SLSQP", options={"ftol": 1e-11, "maxiter": 500})
        w = np.maximum(fit.x, 0)
        w /= w.sum()
        accepted = bool(fit.success and error_s(w) <= 1.1 * bs + 1e-8 and score(w) <= score(initial))
        if not accepted:
            w = initial
        model.mix[c].copy_(torch.from_numpy(w.astype(np.float32)))
        records.append({"column_index": c, "source_reference_MSE": bs, "target_input_reference_MSE": bt,
                        "source_val_MSE": error_s(w), "target_input_pretext_MSE": error_t(w),
                        "selection_score": float(score(w)), "accepted": accepted, "solver": str(fit.message)})
    return records


def run_dataset(ds, args):
    started = time.time()
    data = prepare(ds)
    cfg, src, tgt = data["cfg"], data["source"], data["target"]
    out, dest = RESULTS_DIR / "strict_context" / ds, MODEL_DIR / "strict_context" / ds
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    code_names = ("strict_context.py", "context_models.py", "strict_models.py", "refined_models.py", "refinement.py")
    manifest = {"dataset": ds, "args": vars(args), "torch": torch.__version__, "python": platform.python_version(),
                "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in code_names},
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((PROC_DIR / ds).glob("*.csv"))},
                "target_protocol": "withhold H INSIDE each supplied X; linear branch uses L-H real observations without padding; no cross-window reconstruction; target Y excluded",
                "selection": "per-channel .5 source-relative MSE + .5 target-input-relative MSE, source degradation <=10%; source fallback",
                "evaluation": "reused development benchmark; no new external holdout"}
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != manifest:
        raise ValueError("Manifest differs; use a separate experiment directory")
    write_json(out / "manifest.json", manifest)
    print(f"{ds}: strict label-free source fit", flush=True)
    linear, linear_choices, stats = fit_source(cfg, src, out)
    linear_adapted, adapter_choices = fit_input_only(linear, src, stats, tgt["train"][0], tgt["val"][0], out)
    write_json(out / "linear_selection.json", {"source": linear_choices, "input_only_adapter": adapter_choices})
    frozen = {}
    for seed in args.seeds:
        old = MODEL_DIR / "improved" / ds / f"seed_{seed}"
        old_transfer = MODEL_DIR / "transfer_selection" / ds / f"seed_{seed}"
        selection = json.loads((RESULTS_DIR / "improved" / ds / f"seed_{seed}_selection.json").read_text())
        model = StrictEnsemble(cfg, selection["model_spec"])
        model.models["linear_source"].load_state_dict(linear.state_dict())
        model.models["linear_adapted"].load_state_dict(linear_adapted.state_dict())
        paths = {"neural_source": old / "source_model.pt", "neural_adapted": old / "adapted_model.pt",
                 "dlinear_source": old / "dlinear.pt", "dlinear_adapted": old_transfer / "adapted_model.pt"}
        # transfer_selection's per-seed adapted_model can be a main-family model;
        # select the DLinear adapter solely by the saved within-family proxy.
        history = json.loads((RESULTS_DIR / "transfer_selection" / ds / f"seed_{seed}_selection.json").read_text())
        eligible = [n for n, row in history["scores"].items() if n.startswith("dlinear_") and row["eligible"]]
        chosen = min(eligible, key=lambda n: history["scores"][n]["proxy"]) if eligible else "dlinear_source"
        paths["dlinear_adapted"] = old / "dlinear.pt" if chosen == "dlinear_source" else old_transfer / f"{chosen.removeprefix('dlinear_')}.pt"
        for name, path in paths.items():
            model.models[name].load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        ps = source_calibration(model, src["val"])
        source = copy.deepcopy(model)
        records = strict_calibration(model, ps, src["val"][1], tgt["val"][0])
        choices = {"seed": seed, "components": COMPONENTS, "source_weights": source.mix.tolist(),
                   "adapted_weights": model.mix.tolist(), "calibration": records,
                   "model_spec": {"architecture": "strict_context_ensemble", "neural_spec": selection["model_spec"]},
                   "target_future_labels_used_for_training_or_selection": False,
                   "cross_window_target_label_reconstruction": False}
        seed_dir = dest / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        frozen[seed] = {"source": source.eval(), "adapted": model.eval()}
        for name, m in frozen[seed].items():
            torch.save(m.state_dict(), seed_dir / f"{name}_model.pt")
        write_json(out / f"seed_{seed}_selection.json", choices)
        if seed == args.seeds[0]:
            for name, m in frozen[seed].items():
                torch.save(m.state_dict(), dest / f"{name}_model.pt")
            save_meta(ds, data, dest / "meta.json")
            meta = json.loads((dest / "meta.json").read_text())
            meta.update(model_spec=choices["model_spec"], training_seed=seed,
                        adaptation="strict_input_only", experiment="strict_context")
            write_json(dest / "meta.json", meta)
            write_json(dest / "selection.json", choices)
    write_json(out / "selection_frozen.json", {"test_evaluated_at_freeze": False, "seeds": args.seeds,
               "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    print(f"{ds}: strict choices frozen; evaluating reused development test", flush=True)
    records, examples = [], {}
    for seed, pair in frozen.items():
        for role in ("source", "target"):
            dom = data[role]
            x, y = (v.numpy() for v in dom["test"])
            denom = fixed_scale(ds, role, dom["scaler"])
            models = {"strict_source": pair["source"], "strict_adapted": pair["adapted"],
                      "strict_channel_source": linear, "strict_channel_adapted": linear_adapted,
                      "seasonal_naive": seasonal_naive(x, cfg["H"], cfg["season"]),
                      "persistence": seasonal_naive(x, cfg["H"], 1),
                      "input_mean": np.repeat(x.mean(1, keepdims=True), cfg["H"], axis=1)}
            indices = np.linspace(0, len(x) - 1, min(256, len(x)), dtype=int)
            for method, m in models.items():
                scores, p = evaluate(ds, cfg, m, dom, denom)
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
    pd.concat([pd.read_csv(RESULTS_DIR / "strict_context" / ds / "metrics_all.csv") for ds in args.datasets],
              ignore_index=True).to_csv(RESULTS_DIR / "strict_context" / "metrics_all.csv", index=False)


if __name__ == "__main__":
    main()
