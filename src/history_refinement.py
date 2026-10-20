"""Forecast self-supervision on complete windows inside observed target inputs.

Neural adaptation: target-history MSE + .25 source replay MSE + .001 L2-SP.
Start from original source checkpoints; choose epoch on target observed history,
including epoch zero. A per-variable convex ensemble combines the adapted patch
MLP, DLinear, anchored channel ridge and two simple mean baselines. All selection
is frozen before evaluating the reused development test.
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
from .data import batches, infinite, prepare, save_meta
from .history_models import HistoryEnsemble
from .improve import evaluate, fixed_scale, write_json
from .losses import l2_sp
from .metrics import seasonal_naive
from .refined_models import convex_weights
from .refinement import observed_examples
from .train import predict, set_seed


def adapt_complete_history(source, source_train, history_train, history_val, epochs, seed, log):
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = copy.deepcopy(source)
    reference = {k: v.detach().clone() for k, v in source.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best = float(np.mean((predict(model, history_val[0]) - history_val[1].numpy()) ** 2))
    state, selected = copy.deepcopy(model.state_dict()), 0
    history = [{"epoch": 0, "history_val_MSE": best}]
    source_batches = infinite(*source_train, bs=128, gen=gen)
    for epoch in range(1, epochs + 1):
        model.train()
        totals, count = np.zeros(3), 0
        for xt, yt in batches(*history_train, bs=256, gen=gen):
            xs, ys = next(source_batches)
            target_loss = F.mse_loss(model(xt), yt)
            replay_loss = .25 * F.mse_loss(model(xs), ys)
            anchor_loss = .001 * l2_sp(model, reference)
            loss = target_loss + replay_loss + anchor_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            totals += np.array([float(v.detach()) for v in (target_loss, replay_loss, anchor_loss)]) * len(xt)
            count += len(xt)
        scheduler.step()
        val = float(np.mean((predict(model, history_val[0]) - history_val[1].numpy()) ** 2))
        history.append({"epoch": epoch, "history_val_MSE": val,
                        **dict(zip(("target_history_loss", "source_replay_loss", "anchor_loss"), (totals / count).tolist()))})
        if val < best:
            best, state, selected = val, copy.deepcopy(model.state_dict()), epoch
        log(f"ep {epoch:02d} history_val={val:.6f} best={best:.6f}")
    model.load_state_dict(state)
    return model.eval(), history, selected


@torch.no_grad()
def calibrate(model, x, y):
    components = torch.cat([model.components(xb) for xb in batches(x, bs=256, shuffle=False)]).numpy()
    weights = convex_weights([components[..., i] for i in range(5)], y.numpy())
    model.mix.copy_(torch.from_numpy(weights))
    prediction = (components * weights[None, None]).sum(-1)
    return {"weights_channelridge_patchmlp_dlinear_inputmean_trainingmean": weights.tolist(),
            "validation_MSE": float(np.mean((prediction - y.numpy()) ** 2)),
            "per_channel_validation_MSE": np.mean((prediction - y.numpy()) ** 2, axis=(0, 1)).tolist()}


def run_dataset(ds, args):
    started = time.time()
    data = prepare(ds)
    cfg, src, tgt = data["cfg"], data["source"], data["target"]
    out, dest = RESULTS_DIR / "history_refined" / ds, MODEL_DIR / "history_refined" / ds
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": ds, "args": vars(args), "torch": torch.__version__, "python": platform.python_version(),
                "loss": {"target_history": 1., "source_replay": .25, "l2_sp": .001},
                "optimizer": {"name": "AdamW", "lr": .0003, "weight_decay": .0001, "schedule": "cosine"},
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((PROC_DIR / ds).glob("*.csv"))},
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__), Path(__file__).with_name("history_models.py"),
                                 Path(__file__).with_name("refined_models.py"), Path(__file__).with_name("refinement.py"))},
                "protocol": "fit and validate forecasts within observed target X union, not original target Y tensors",
                "evaluation": "reused development test, not external confirmation"}
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != manifest:
        raise ValueError("Manifest differs; do not overwrite an existing run")
    write_json(out / "manifest.json", manifest)
    history_train, audit_train = observed_examples(ds, tgt["train"][0], "train")
    history_val, audit_val = observed_examples(ds, tgt["val"][0], "val")
    write_json(out / "history_audit.json", {"train": audit_train, "validation": audit_val})
    frozen = {}
    for seed in args.seeds:
        seed_dir = dest / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        old = MODEL_DIR / "improved" / ds / f"seed_{seed}"
        spec = json.loads((RESULTS_DIR / "improved" / ds / f"seed_{seed}_selection.json").read_text())["model_spec"]
        source, adapted = HistoryEnsemble(cfg, spec), HistoryEnsemble(cfg, spec)
        source.linear.load_state_dict(torch.load(MODEL_DIR / "refined" / ds / "source_linear.pt", weights_only=True))
        adapted.linear.load_state_dict(torch.load(MODEL_DIR / "refined" / ds / "adapted_linear.pt", weights_only=True))
        source.neural.load_state_dict(torch.load(old / "source_model.pt", weights_only=True))
        source.dlinear.load_state_dict(torch.load(old / "dlinear.pt", weights_only=True))
        choices = {"seed": seed, "model_spec": {"architecture": "history_ensemble", "neural_spec": spec},
                   "selection_used_test_futures": False}
        def log(message):
            print(f"{ds} s{seed} {message}", flush=True)
            with (out / f"seed_{seed}.log").open("a", encoding="utf-8") as f:
                f.write(message + "\n")
        for name in ("neural", "dlinear"):
            checkpoint, histfile = seed_dir / f"{name}.pt", out / f"seed_{seed}_{name}.json"
            if checkpoint.exists() and histfile.exists():
                model = copy.deepcopy(getattr(source, name))
                model.load_state_dict(torch.load(checkpoint, weights_only=True))
                history = json.loads(histfile.read_text())
                epoch = min(history, key=lambda row: row["history_val_MSE"])["epoch"]
            else:
                log(f"adapting {name} from source")
                model, history, epoch = adapt_complete_history(getattr(source, name), src["train"],
                    history_train, history_val, args.epochs, seed, lambda msg: log(name + " " + msg))
                torch.save(model.state_dict(), checkpoint)
                write_json(histfile, history)
            setattr(adapted, name, model)
            choices[f"{name}_selected_epoch"] = epoch
        choices["source"] = calibrate(source.eval(), *src["val"])
        choices["adapted"] = calibrate(adapted.eval(), *history_val)
        for name, model in (("source", source), ("adapted", adapted)):
            torch.save(model.state_dict(), seed_dir / f"{name}_model.pt")
        frozen[seed] = {"source": source.eval(), "adapted": adapted.eval()}
        write_json(out / f"seed_{seed}_selection.json", choices)
        if seed == args.seeds[0]:
            for name, model in frozen[seed].items():
                torch.save(model.state_dict(), dest / f"{name}_model.pt")
            save_meta(ds, data, dest / "meta.json")
            meta = json.loads((dest / "meta.json").read_text())
            meta.update(model_spec=choices["model_spec"], training_seed=seed,
                        adaptation="complete_history_neural_ridge", experiment="history_refined")
            write_json(dest / "meta.json", meta)
            write_json(dest / "selection.json", choices)
    write_json(out / "selection_frozen.json", {"test_evaluated_at_freeze": False, "seeds": args.seeds,
               "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    print(f"{ds}: choices frozen; evaluating reused development test", flush=True)
    records, examples = [], {}
    for seed, pair in frozen.items():
        for role in ("source", "target"):
            dom = data[role]
            denom = fixed_scale(ds, role, dom["scaler"])
            x, y = (v.numpy() for v in dom["test"])
            models = {"history_source": pair["source"], "history_adapted": pair["adapted"],
                      "history_patchmlp": pair["adapted"].neural, "history_dlinear": pair["adapted"].dlinear,
                      "seasonal_naive": seasonal_naive(x, cfg["H"], cfg["season"]),
                      "persistence": seasonal_naive(x, cfg["H"], 1),
                      "input_mean": np.repeat(x.mean(1, keepdims=True), cfg["H"], axis=1)}
            indices = np.linspace(0, len(x) - 1, min(256, len(x)), dtype=int)
            for name, model in models.items():
                row, p = evaluate(ds, cfg, model, dom, denom)
                records.append({"dataset": ds, "seed": seed, "domain": role, "method": name, **row})
                if seed == args.seeds[0]:
                    examples[f"{role}_{name}"] = p[indices]
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
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for ds in args.datasets:
        run_dataset(ds, args)
    pd.concat([pd.read_csv(RESULTS_DIR / "history_refined" / ds / "metrics_all.csv")
               for ds in args.datasets], ignore_index=True).to_csv(RESULTS_DIR / "history_refined" / "metrics_all.csv", index=False)


if __name__ == "__main__":
    main()
