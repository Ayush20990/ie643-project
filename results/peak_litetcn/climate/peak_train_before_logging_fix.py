"""Peak-focused LiteTCN revision with strictly within-input target adaptation.

Source ablations: no clock/MSE, clock/MSE, clock/ramp/extrema. All use the same
training budget. A source-validation score fixes the winning LiteTCN variant
before adapting/evaluating it. Target selection sees only each window's prefix
and its withheld observed tail. No target future-label tensor is accepted.
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
from .data import batches, infinite, save_meta
from .diagnose_peaks import peak_metrics
from .improve import evaluate, fixed_scale, write_json
from .losses import l2_sp
from .peak_data import prepare_peak
from .peak_model import PeakLiteTCN, forecast_loss, pretext
from .train import predict, set_seed

CANDIDATES = {"no_clock_mse": {"calendar": False, "shape_loss": False},
              "clock_mse": {"calendar": True, "shape_loss": False},
              "clock_shape": {"calendar": True, "shape_loss": True}}


def make_peak(cfg, spec):
    return PeakLiteTCN(cfg, **{k: v for k, v in spec.items() if k not in ("shape_loss", "architecture")})


def quality(prediction, truth, key):
    y, p = np.asarray(truth), np.asarray(prediction)
    error = (p-y)**2
    j = np.arange(len(y))
    actual_high, actual_low = y[..., key].argmax(1), y[..., key].argmin(1)
    peak = .5 * (error[j, actual_high, key].mean() + error[j, actual_low, key].mean())
    ramp = np.mean((np.diff(p[..., key], axis=1) - np.diff(y[..., key], axis=1))**2)
    mse, key_mse = error.mean(), error[..., key].mean()
    return {"MSE": float(mse), "key_MSE": float(key_mse), "key_peak_trough_MSE": float(peak),
            "key_ramp_MSE": float(ramp), "score": float(.5*mse + .5*key_mse + .25*peak + .1*ramp)}


def validation(model, data, key):
    return quality(predict(model, data[0], bs=256), data[1].numpy(), key)


def input_validation(model, inputs, cfg):
    prefix, observed = pretext(inputs, cfg["H"], len(cfg["columns"]))
    return validation(model, (prefix, observed), cfg["columns"].index(cfg["key"]))


def fit_source(model, train, val, cfg, shape, args, seed, log):
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    key = cfg["columns"].index(cfg["key"])
    best = validation(model, val, key)
    state, selected = copy.deepcopy(model.state_dict()), 0
    history = [{"round": 0, **best}]
    x, y = train
    for epoch in range(1, args.epochs+1):
        model.train()
        # Fresh uniformly sampled windows each round, without replacement.
        indices = torch.randperm(len(x), generator=generator)[:args.max_windows]
        total, count = 0., 0
        for start in range(0, len(indices), args.batch_size):
            ix = indices[start:start+args.batch_size]
            loss = forecast_loss(model(x[ix]), y[ix], key, shape)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach())*len(ix)
            count += len(ix)
        scheduler.step()
        measured = validation(model, val, key)
        history.append({"round": epoch, "sampled_windows": count, "train_loss": total/count, **measured})
        if measured["score"] < best["score"]:
            best, state, selected = measured, copy.deepcopy(model.state_dict()), epoch
        log(f"round {epoch:02d} val_score={measured['score']:.6f} key_MSE={measured['key_MSE']:.6f}")
    model.load_state_dict(state)
    return model.eval(), history, best, selected


def adapt_input_only(source, source_train, source_val, target_train_inputs, target_val_inputs,
                     cfg, shape, args, seed, log):
    """Target Y is deliberately absent from this API; pretexts stay inside X."""
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = copy.deepcopy(source)
    reference = {name: p.detach().clone() for name, p in source.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.adapt_epochs)
    key = cfg["columns"].index(cfg["key"])
    base_s, base_t = validation(source, source_val, key), input_validation(source, target_val_inputs, cfg)
    state, best, selected = copy.deepcopy(source.state_dict()), 1., 0
    history = [{"round": 0, "source": base_s, "target_input": base_t, "proxy": 1., "eligible": True}]
    replay = infinite(*source_train, bs=args.batch_size//2, gen=generator)
    for epoch in range(1, args.adapt_epochs+1):
        model.train()
        indices = torch.randperm(len(target_train_inputs), generator=generator)[:args.max_windows]
        totals, count = np.zeros(3), 0
        for start in range(0, len(indices), args.batch_size):
            xt = target_train_inputs[indices[start:start+args.batch_size]]
            prefix, observed = pretext(xt, cfg["H"], len(cfg["columns"]))
            xs, ys = next(replay)
            losses = (forecast_loss(model(prefix), observed, key, shape),
                      .25*forecast_loss(model(xs), ys, key, shape), .001*l2_sp(model, reference))
            loss = sum(losses)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            totals += np.array([float(v.detach()) for v in losses])*len(xt)
            count += len(xt)
        schedule.step()
        sv, tv = validation(model, source_val, key), input_validation(model, target_val_inputs, cfg)
        proxy = .5*sv["score"]/base_s["score"] + .5*tv["score"]/base_t["score"]
        eligible = sv["MSE"] <= 1.1*base_s["MSE"]
        history.append({"round": epoch, "source": sv, "target_input": tv, "proxy": proxy,
                        "eligible": eligible, **dict(zip(("input_loss", "replay_loss", "anchor_loss"), (totals/count).tolist()))})
        if eligible and proxy < best:
            best, state, selected = proxy, copy.deepcopy(model.state_dict()), epoch
        log(f"adapt {epoch:02d} proxy={proxy:.6f} eligible={eligible} input_key_MSE={tv['key_MSE']:.6f}")
    model.load_state_dict(state)
    return model.eval(), history, best, selected


def run_dataset(ds, args):
    data = prepare_peak(ds)
    cfg, src, tgt = data["cfg"], data["source"], data["target"]
    out, dest = RESULTS_DIR/"peak_litetcn"/ds, MODEL_DIR/"peak_litetcn"/ds
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    code_names = ("peak_train.py", "peak_model.py", "peak_data.py", "data.py", "losses.py", "metrics.py")
    manifest = {"dataset": ds, "args": vars(args), "torch": torch.__version__, "python": platform.python_version(),
                "code_sha256": {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in code_names},
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((PROC_DIR/ds).glob("*.csv"))},
                "candidates": CANDIDATES, "parameters": "seasonal-folded depthwise dilated LiteTCN, width 32, 4 blocks",
                "source_selection": ".5 all-variable MSE + .5 key MSE + .25 key peak/trough MSE + .1 key ramp MSE",
                "target_protocol": "within EACH X: forecast final H observed values from real L-H prefix and deterministic clock; no target Y, no cross-window reconstruction",
                "adaptation": {"source_replay": .25, "L2SP": .001, "source_MSE_guard": 1.1},
                "evaluation": "reused development period; previous 2025 period also already observed, not fresh for this round"}
    if (out/"manifest.json").exists() and json.loads((out/"manifest.json").read_text()) != manifest:
        raise ValueError("Manifest differs; preserve old run in a separate directory")
    write_json(out/"manifest.json", manifest)
    def log(message):
        print(f"{ds} {message}", flush=True)
        with (out/"run.log").open("a", encoding="utf-8") as f:
            f.write(message+"\n")
    source_scores, first_models = {}, {}
    first_seed = args.seeds[0]
    for name, spec in CANDIDATES.items():
        checkpoint = dest/f"screen_{first_seed}_{name}.pt"
        record = out/f"screen_{first_seed}_{name}.json"
        set_seed(first_seed)
        model = make_peak(cfg, spec)
        if checkpoint.exists() and record.exists():
            saved = json.loads(record.read_text())
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            log(f"resumed {name}")
        else:
            log(f"source screen {name}, parameters={sum(p.numel() for p in model.parameters())}")
            model, history, best, epoch = fit_source(model, src["train_packed"], src["val_packed"], cfg,
                spec["shape_loss"], args, first_seed, lambda m: log(name+" "+m))
            saved = {"history": history, "best": best, "selected_round": epoch}
            torch.save(model.state_dict(), checkpoint)
            write_json(record, saved)
        source_scores[name], first_models[name] = saved["best"]["score"], model
    chosen = min(source_scores, key=source_scores.get)
    spec = CANDIDATES[chosen]
    write_json(out/"source_architecture_frozen.json", {"chosen": chosen, "source_scores": source_scores,
               "target_test_evaluated": False, "model_spec": {"architecture": "peak_litetcn", **spec}})
    log(f"source-validation choice: {chosen}")
    frozen = {}
    for seed in args.seeds:
        seed_dir = dest/f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        source = copy.deepcopy(first_models[chosen]) if seed == first_seed else make_peak(cfg, spec)
        checkpoint, history_file = seed_dir/"source_model.pt", out/f"seed_{seed}_source.json"
        if seed != first_seed:
            if checkpoint.exists() and history_file.exists():
                source.load_state_dict(torch.load(checkpoint, weights_only=True))
            else:
                set_seed(seed)
                source = make_peak(cfg, spec)
                source, history, best, epoch = fit_source(source, src["train_packed"], src["val_packed"], cfg,
                    spec["shape_loss"], args, seed, lambda m: log(f"s{seed} source "+m))
                write_json(history_file, {"history": history, "best": best, "selected_round": epoch})
        else:
            write_json(history_file, json.loads((out/f"screen_{first_seed}_{chosen}.json").read_text()))
        torch.save(source.state_dict(), checkpoint)
        models, proxies, epochs = {"source_only": source}, {"source_only": 1.}, {"source_only": 0}
        for name, shape in (("input_mse", False), ("input_shape", True)):
            checkpoint, record = seed_dir/f"{name}.pt", out/f"seed_{seed}_{name}.json"
            if checkpoint.exists() and record.exists():
                model = make_peak(cfg, spec)
                model.load_state_dict(torch.load(checkpoint, weights_only=True))
                history = json.loads(record.read_text())
                best_row = min((r for r in history if r["eligible"]), key=lambda r: r["proxy"])
                score, epoch = best_row["proxy"], best_row["round"]
            else:
                model, history, score, epoch = adapt_input_only(source, src["train_packed"], src["val_packed"],
                    tgt["train_packed"][0], tgt["val_packed"][0], cfg, shape, args, seed,
                    lambda m: log(f"s{seed} {name} "+m))
                torch.save(model.state_dict(), checkpoint)
                write_json(record, history)
            models[name], proxies[name], epochs[name] = model.eval(), score, epoch
        recipe = min(proxies, key=proxies.get)
        selected = models[recipe]
        torch.save(selected.state_dict(), seed_dir/"adapted_model.pt")
        choices = {"seed": seed, "source_variant": chosen, "model_spec": {"architecture": "peak_litetcn", **spec},
                   "adaptation": recipe, "proxy": proxies, "selected_rounds": epochs,
                   "target_future_labels_used_for_training_or_selection": False}
        write_json(out/f"seed_{seed}_selection.json", choices)
        frozen[seed] = models
        models["selected_adapted"] = selected
        if seed == first_seed:
            torch.save(source.state_dict(), dest/"source_model.pt")
            torch.save(selected.state_dict(), dest/"adapted_model.pt")
            save_meta(ds, data, dest/"meta.json")
            meta = json.loads((dest/"meta.json").read_text())
            meta.update(model_spec=choices["model_spec"], training_seed=seed, adaptation=recipe,
                        experiment="peak_litetcn", requires_dates=spec["calendar"], effective_context=cfg["L"]-cfg["H"])
            write_json(dest/"meta.json", meta)
            write_json(dest/"selection.json", choices)
    write_json(out/"selection_frozen.json", {"test_evaluated_at_freeze": False, "seeds": args.seeds,
               "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    log("all choices frozen; evaluate reused development test")
    rows, peak_rows, examples = [], [], {}
    for seed, models in frozen.items():
        for role in ("source", "target"):
            dom = data[role]
            denom = fixed_scale(ds, role, dom["scaler"])
            x, y = dom["test"]
            key = cfg["columns"].index(cfg["key"])
            indices = np.linspace(0, len(x)-1, min(256, len(x)), dtype=int)
            for name, model in models.items():
                p = predict(model, dom["test_packed"][0], bs=256)
                metrics, _ = evaluate(ds, cfg, p, dom, denom)
                rows.append({"dataset": ds, "seed": seed, "domain": role, "method": name, **metrics})
                sd, mu = dom["scaler"].std[key], dom["scaler"].mean[key]
                peak_rows.append({"dataset": ds, "seed": seed, "domain": role, "method": name,
                                  **peak_metrics(y.numpy()[..., key]*sd+mu, p[..., key]*sd+mu)})
                if seed == first_seed:
                    examples[f"{role}_{name}"] = p[indices]
            if seed == first_seed:
                examples[f"{role}_X"], examples[f"{role}_Y"] = x.numpy()[indices], y.numpy()[indices]
                examples[f"{role}_indices"] = indices
    # Source ablation tests are descriptive only; architecture choice is already frozen.
    for name, model in first_models.items():
        for role in ("source", "target"):
            p = predict(model, data[role]["test_packed"][0], bs=256)
            metrics, _ = evaluate(ds, cfg, p, data[role], fixed_scale(ds, role, data[role]["scaler"]))
            rows.append({"dataset": ds, "seed": first_seed, "domain": role, "method": f"screen_{name}", **metrics})
    np.savez_compressed(out/"forecast_examples.npz", **examples)
    pd.DataFrame(rows).to_csv(out/"metrics_all.csv", index=False)
    pd.DataFrame(peak_rows).to_csv(out/"peak_metrics.csv", index=False)
    write_json(out/"timing.json", {"seconds": time.time()-started})
    log(pd.DataFrame(rows).query("domain == 'target'").groupby("method")[["MSE", "key_RMSE"]].mean().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("climate", "energy"), default=["climate", "energy"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--adapt-epochs", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for ds in args.datasets:
        run_dataset(ds, args)


if __name__ == "__main__":
    main()
