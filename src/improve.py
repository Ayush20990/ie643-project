"""Reproducible architecture sweep and input-only adaptation, separate from legacy artifacts.

python -m src.improve --datasets climate energy finance --seeds 0 1 2
Selection uses source validation futures and target validation INPUTS only. Test
forecasts are generated after source architecture, adaptation recipe and epoch freeze.
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
import torch.nn.functional as F

from .config import DATASETS, MODEL_DIR, PROC_DIR, RESULTS_DIR
from .data import batches, infinite, load_domain, prepare, save_meta, series_list, chrono_split
from .losses import l2_sp, mmd_rbf
from .metrics import all_metrics, seasonal_naive
from .model import make_forecaster, n_params
from .train import predict, set_seed, train_supervised

CANDIDATES = {
    "tcn3": {"architecture": "litetcn", "depth": 3},
    "tcn5": {"architecture": "litetcn", "depth": 5},
    "tcn7": {"architecture": "litetcn", "depth": 7},
    "patch_mlp": {"architecture": "patch_mlp"},
    "dlinear": {"architecture": "dlinear"},
}
RECIPES = {"replay_inner": {"inner": 0.5, "anchor": 0.001},
           "replay_inner_mmd": {"inner": 0.5, "mmd": 0.05, "anchor": 0.001}}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


@torch.no_grad()
def inner_score(model, x):
    model.eval()
    total = sum(float(model.inner_forecast_loss(xb)) * len(xb)
                for xb in batches(x, bs=256, shuffle=False))
    return total / len(x)


@torch.no_grad()
def validation_mse(model, domain):
    return float(np.mean((predict(model, domain["val"][0]) - domain["val"][1].numpy()) ** 2))


def adapt_history(source, src, tgt_inputs, probe, recipe, epochs, seed, log):
    """No target future tensor is accepted by this function.

    Objective: source forecast MSE + .5 target-history forecast MSE + .001
    L2-SP, optionally .05 MMD. The last H observed input steps are withheld,
    with constant left padding of the prefix to preserve the model input size.
    Checkpoint proxy = .5 source-val/base + .5 target-input-pretext/base.
    Include epoch zero; reject checkpoints increasing source-val MSE by >10%.
    """
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = copy.deepcopy(source)
    ref = {k: v.detach().clone() for k, v in source.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    base_src = validation_mse(source, src)
    base_inner = inner_score(source, probe)
    best, best_state, chosen_epoch = 1.0, copy.deepcopy(source.state_dict()), 0
    history = [{"epoch": 0, "source_val_mse": base_src, "target_input_mse": base_inner,
                "selection_score": 1.0, "eligible": True}]
    it_s = infinite(*src["train"], bs=128, gen=gen)
    it_t = infinite(tgt_inputs, bs=128, gen=gen)
    steps = int(np.ceil(len(tgt_inputs) / 256))
    for epoch in range(1, epochs + 1):
        model.train()
        acc = {}
        for _ in range(steps):
            xs, ys = next(it_s)
            xt = next(it_t)
            ps, zs = model(xs, return_z=True)
            terms = {"source": F.mse_loss(ps, ys),
                     "inner": recipe["inner"] * model.inner_forecast_loss(xt),
                     "anchor": recipe["anchor"] * l2_sp(model, ref)}
            if recipe.get("mmd"):
                zt = model(xt, return_z=True)[1]
                terms["mmd"] = recipe["mmd"] * mmd_rbf(zs, zt)
            loss = sum(terms.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for key, value in terms.items():
                acc[key] = acc.get(key, 0.0) + float(value.detach()) / steps
        sv, tv = validation_mse(model, src), inner_score(model, probe)
        score = .5 * sv / max(base_src, 1e-8) + .5 * tv / max(base_inner, 1e-8)
        eligible = sv <= 1.1 * base_src
        history.append({"epoch": epoch, **acc, "source_val_mse": sv,
                        "target_input_mse": tv, "selection_score": score, "eligible": eligible})
        log(f"adapt ep={epoch} source_val={sv:.5f} input_pretext={tv:.5f} proxy={score:.5f}")
        if eligible and score < best:
            best, best_state, chosen_epoch = score, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    return model, history, best, chosen_epoch


def fixed_scale(ds, role, scaler):
    cfg = DATASETS[ds]
    arrays = series_list(load_domain(ds, cfg[role]), cfg["columns"])
    train = chrono_split(arrays, cfg["split"], cfg["L"])["train"]
    differences = [np.abs(a[cfg["season"]:] - a[:-cfg["season"]]) for a in train]
    return np.concatenate(differences).mean(0) / scaler.std


def evaluate(ds, cfg, model, dom, denom):
    x, y = (v.numpy() for v in dom["test"])
    p = model if isinstance(model, np.ndarray) else predict(model, dom["test"][0], bs=256)
    row = all_metrics(x, y, p, cfg["season"], dom["scaler"], cfg["columns"].index(cfg["key"]))
    per_mae = np.abs(y - p).mean((0, 1))
    row["MASE_train"] = float(np.mean(per_mae / np.maximum(denom, 1e-8)))
    for i, name in enumerate(cfg["columns"]):
        row[f"RMSE_{name}"] = float(np.sqrt(np.mean((y[..., i] - p[..., i]) ** 2)) * dom["scaler"].std[i])
    return row, p


def run_dataset(ds, args):
    data = prepare(ds)
    cfg, src, tgt = data["cfg"], data["source"], data["target"]
    out = RESULTS_DIR / "improved" / ds
    dest = MODEL_DIR / "improved" / ds
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": ds, "args": vars(args), "torch": torch.__version__,
                "python": platform.python_version(), "candidate_specs": CANDIDATES,
                "recipes": RECIPES, "source_ssl_weight": 0,
                "selection": "source validation MSE; adaptation input-only proxy with epoch-zero fallback",
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((PROC_DIR / ds).glob("*.csv"))}}
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Run settings differ from {manifest_path}; use a separate output or restore settings")
    write_json(manifest_path, manifest)
    denominators = {r: fixed_scale(ds, r, data[r]["scaler"]) for r in ("source", "target")}
    for seed in args.seeds:
        result_file = out / f"seed_{seed}.csv"
        if result_file.exists():
            print(f"{ds} seed {seed}: already complete", flush=True)
            continue
        started = time.time()
        seed_dir = dest / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        log_path = out / f"seed_{seed}.log"
        def log(message):
            print(f"{ds} s{seed} {message}", flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        trained, histories, scores = {}, {}, {}
        for name, spec in CANDIDATES.items():
            set_seed(seed)
            model = make_forecaster(cfg, spec)
            checkpoint = seed_dir / f"{name}.pt"
            histpath = out / f"seed_{seed}_{name}.json"
            if checkpoint.exists() and histpath.exists():
                model.load_state_dict(torch.load(checkpoint, weights_only=True))
                hist = json.loads(histpath.read_text())
                log(f"resumed {name}")
            else:
                log(f"training {name}, parameters={n_params(model):,}")
                model, hist = train_supervised(model, src["train"], src["val"],
                    epochs=args.epochs, ssl=0, seed=seed, log=log, tag=name)
                torch.save(model.state_dict(), checkpoint)
                write_json(histpath, hist)
            trained[name], histories[name] = model, hist
            scores[name] = validation_mse(model, src)
        chosen = min(scores, key=scores.get)
        source = trained[chosen]
        log(f"source-validation architecture choice: {chosen} ({scores[chosen]:.5f})")
        # Only inputs from target validation are supplied to model selection.
        probe = tgt["val"][0][::max(1, len(tgt["val"][0]) // 512)][:512]
        adapted = {}
        adapt_scores = {"source_only": 1.0}
        epochs_selected = {"source_only": 0}
        for name, recipe in RECIPES.items():
            model, hist, score, ep = adapt_history(source, src, tgt["train"][0], probe,
                recipe, args.adapt_epochs, seed, lambda msg: log(name + " " + msg))
            adapted[name], adapt_scores[name], epochs_selected[name] = model, score, ep
            write_json(out / f"seed_{seed}_{name}.json", hist)
            torch.save(model.state_dict(), seed_dir / f"{name}.pt")
        adapted_name = min(adapt_scores, key=adapt_scores.get)
        selected_model = source if adapted_name == "source_only" else adapted[adapted_name]
        selection = {"seed": seed, "architecture": chosen, "model_spec": CANDIDATES[chosen],
                     "source_val_mse": scores, "adaptation": adapted_name,
                     "adaptation_proxy": adapt_scores, "epochs_selected": epochs_selected}
        write_json(out / f"seed_{seed}_selection.json", selection)
        # Freeze the deployment choice BEFORE any target test scores are calculated.
        torch.save(source.state_dict(), seed_dir / "source_model.pt")
        torch.save(selected_model.state_dict(), seed_dir / "adapted_model.pt")
        if seed == args.seeds[0]:
            torch.save(source.state_dict(), dest / "source_model.pt")
            torch.save(selected_model.state_dict(), dest / "adapted_model.pt")
            save_meta(ds, data, dest / "meta.json")
            meta = json.loads((dest / "meta.json").read_text())
            meta.update(model_spec=CANDIDATES[chosen], adaptation=adapted_name,
                        training_seed=seed, experiment="improved")
            write_json(dest / "meta.json", meta)
            write_json(dest / "selection.json", selection)
        # Supervised reference explicitly uses target training/validation futures.
        oracle, hist = train_supervised(copy.deepcopy(source), tgt["train"], tgt["val"],
            epochs=args.epochs, ssl=0, lr=3e-4, seed=seed, log=log, tag="supervised_reference")
        torch.save(oracle.state_dict(), seed_dir / "supervised_reference.pt")
        write_json(out / f"seed_{seed}_supervised_reference.json", hist)
        legacy = make_forecaster(cfg)
        legacy.load_state_dict(torch.load(MODEL_DIR / ds / "source_model.pt", weights_only=True))
        legacy_adapted = make_forecaster(cfg)
        legacy_adapted.load_state_dict(torch.load(MODEL_DIR / ds / "adapted_model.pt", weights_only=True))
        models = {**trained, "selected_source": source, **adapted, "selected_adapted": selected_model,
                  "legacy_source_seed0": legacy, "legacy_care_v2_seed0": legacy_adapted,
                  "supervised_reference": oracle}
        records, forecasts = [], {}
        for role in ("source", "target"):
            dom = data[role]
            x = dom["test"][0].numpy()
            baselines = {"seasonal_naive": seasonal_naive(x, cfg["H"], cfg["season"]),
                         "persistence": seasonal_naive(x, cfg["H"], 1),
                         "input_mean": np.repeat(x.mean(1, keepdims=True), cfg["H"], axis=1)}
            for name, model in {**models, **baselines}.items():
                metrics, p = evaluate(ds, cfg, model, dom, denominators[role])
                records.append({"dataset": ds, "seed": seed, "method": name, "domain": role,
                                "params": n_params(model) if isinstance(model, torch.nn.Module) else 0,
                                **metrics})
                if seed == args.seeds[0] and name in ("selected_source", "selected_adapted",
                        "legacy_care_v2_seed0", "seasonal_naive", "supervised_reference"):
                    # Evenly spaced, deterministic forecast origins; all windows contribute to scores.
                    indices = np.linspace(0, len(x) - 1, min(256, len(x)), dtype=int)
                    forecasts[f"{role}_{name}"] = p[indices]
            if seed == args.seeds[0]:
                forecasts[f"{role}_X"] = x[indices]
                forecasts[f"{role}_Y"] = dom["test"][1].numpy()[indices]
                forecasts[f"{role}_indices"] = indices
        if forecasts:
            np.savez_compressed(out / "forecast_examples.npz", **forecasts)
        pd.DataFrame(records).to_csv(result_file, index=False)
        log(f"completed in {time.time() - started:.1f}s; selected {chosen}/{adapted_name}")
    frames = [pd.read_csv(out / f"seed_{s}.csv") for s in args.seeds if (out / f"seed_{s}.csv").exists()]
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out / "metrics_all.csv", index=False)
        cols = ["MSE", "MAE", "MASE_train", "key_RMSE", "params"]
        summary = df.groupby(["domain", "method"])[cols].agg(["mean", "std", "count"])
        summary.columns = ["_".join(col) for col in summary.columns]
        summary.to_csv(out / "summary.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--adapt-epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.epochs < 1 or args.adapt_epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        ap.error("epochs must be positive and seeds unique")
    torch.set_num_threads(args.threads)
    for ds in args.datasets:
        run_dataset(ds, args)


if __name__ == "__main__":
    main()
