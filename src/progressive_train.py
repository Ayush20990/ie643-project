"""Matched head-only vs gradual-unfreezing control, strict input-only adaptation.

All seeds/controls are fitted and selected before any test forecasting. Previously
inspected tests are development evaluations, not independent confirmation.
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

from .config import MODEL_DIR, PROC_DIR, RESULTS_DIR
from .data import batches, prepare, save_meta
from .diagnose_peaks import peak_metrics
from .improve import evaluate, fixed_scale, write_json
from .progressive_tcn import ProgressiveLiteTCN, forecast_objective, input_pair, relative_anchor
from .train import predict, set_seed

FAMILY = "progressive_litetcn"
MODES = ("head_only", "progressive")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def quality(model, x, y, key):
    p = predict(model, x, bs=256)
    e = (p-y.numpy())**2
    mse, kmse = float(e.mean()), float(e[..., key].mean())
    return {"MSE": mse, "key_MSE": kmse, "score": .5*mse+.5*kmse}


@torch.no_grad()
def input_quality(model, x, horizon, key):
    # Evaluation still handles each window independently; no assembled target Y.
    total = {k: 0. for k in ("MSE", "key_MSE", "score")}
    for xb in batches(x, bs=256, shuffle=False):
        prefix, tail = input_pair(xb, horizon)
        row = quality(model, prefix, tail, key)
        for name in total:
            total[name] += row[name]*len(xb)/len(x)
    return total


def source_fit(initial, train, val, cfg, mode, seed, args, log):
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed+100)
    model = copy.deepcopy(initial)
    reference = {n: p.detach().clone() for n, p in initial.named_parameters()}
    opt = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.rounds, eta_min=1e-6)
    key = cfg["columns"].index(cfg["key"])
    baseline = quality(model, *val, key)
    best, chosen = copy.deepcopy(model.state_dict()), 0
    best_score = baseline["score"]
    history = [{"round": 0, "stage": "initial_ridge", **baseline, "eligible": True}]
    for r in range(1, args.rounds+1):
        model.train()
        stage = model.set_stage(mode, r)
        indices = torch.randperm(len(train[0]), generator=gen)[:args.max_windows]
        total = 0.
        for i in indices.split(args.batch_size):
            x, y = train[0][i], train[1][i]
            loss = forecast_objective(model(x), y, key)+.001*relative_anchor(model, reference)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach())*len(i)/len(indices)
        sched.step()
        row = quality(model, *val, key)
        eligible = row["MSE"] <= 1.03*baseline["MSE"]
        history.append({"round": r, "stage": stage, "train_loss": total, **row, "eligible": eligible})
        if eligible and row["score"] < best_score:
            best, best_score, chosen = copy.deepcopy(model.state_dict()), row["score"], r
        log(f"{mode} round={r:02d} stage={stage} source_val={row['MSE']:.6f} key={row['key_MSE']:.6f} selected={chosen}")
    model.load_state_dict(best)
    return model.eval(), {"selected_round": chosen, "score": best_score, "history": history}


def adapt_fit(source, source_train, source_val, target_train_x, target_val_x, cfg, seed, args, log):
    """Signature deliberately excludes target forecasting labels."""
    set_seed(seed+1000)
    gen = torch.Generator().manual_seed(seed+1100)
    model = copy.deepcopy(source)
    ref = {n: p.detach().clone() for n, p in source.named_parameters()}
    opt = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4)
    # More conservative rates for adaptation than source fine-tuning.
    for group in opt.param_groups:
        group["lr"] *= .5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.adapt_rounds, eta_min=1e-6)
    key, h = cfg["columns"].index(cfg["key"]), cfg["H"]
    bs, bt = quality(model, *source_val, key), input_quality(model, target_val_x, h, key)
    history = [{"round": 0, "stage": "source_fallback", "source": bs, "target_input": bt, "proxy": 1., "eligible": True}]
    best, best_score, chosen = copy.deepcopy(model.state_dict()), 1., 0
    for r in range(1, args.adapt_rounds+1):
        model.train()
        # Head first, projection next, then the full encoder. Skip/RevIN never change.
        schedule_round = 1 if r <= 2 else 5 if r <= 4 else 9
        stage = model.set_stage("progressive", schedule_round)
        ti = torch.randperm(len(target_train_x), generator=gen)[:args.max_windows]
        si = torch.randint(len(source_train[0]), (len(ti),), generator=gen)
        totals = {"target_input_loss": 0., "source_replay_loss": 0., "anchor": 0.}
        for it, ix in zip(ti.split(args.batch_size), si.split(args.batch_size)):
            prefix, tail = input_pair(target_train_x[it], h)
            target_loss = forecast_objective(model(prefix), tail, key)
            replay = forecast_objective(model(source_train[0][ix]), source_train[1][ix], key)
            anchor = relative_anchor(model, ref)
            loss = target_loss+replay+.01*anchor
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            for name, term in (("target_input_loss", target_loss), ("source_replay_loss", replay), ("anchor", anchor)):
                totals[name] += float(term.detach())*len(it)/len(ti)
        sched.step()
        sv, tv = quality(model, *source_val, key), input_quality(model, target_val_x, h, key)
        proxy = .5*sv["score"]/max(bs["score"], 1e-8)+.5*tv["score"]/max(bt["score"], 1e-8)
        eligible = sv["MSE"] <= 1.1*bs["MSE"]
        history.append({"round": r, "stage": stage, **totals, "source": sv, "target_input": tv, "proxy": proxy, "eligible": eligible})
        if eligible and proxy < best_score:
            best, best_score, chosen = copy.deepcopy(model.state_dict()), proxy, r
        log(f"adapt round={r} stage={stage} source_val={sv['MSE']:.6f} input={tv['MSE']:.6f} proxy={proxy:.6f} selected={chosen}")
    model.load_state_dict(best)
    return model.eval(), {"selected_round": chosen, "proxy": best_score, "history": history,
                          "target_future_labels_used_for_training_or_selection": False}


def fit_dataset(ds, args):
    started = time.perf_counter()
    data = prepare(ds)
    cfg, src, tgt = data["cfg"], data["source"], data["target"]
    out, dest = RESULTS_DIR/FAMILY/ds, MODEL_DIR/FAMILY/ds
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    init_files = [MODEL_DIR/"residual_litetcn"/ds/f"seed_{s}"/"source_model.pt" for s in args.seeds]
    manifest = {"dataset": ds, "args": vars(args), "torch": torch.__version__, "python": platform.python_version(),
        "initialization": "source-only ridge LiteTCN-R, matched within seed", "source_controls": MODES,
        "source_objective": "0.5 global MSE + 0.5 key MSE + 0.001 relative layer anchor",
        "source_selection": "minimum source validation objective, include initial state; global MSE guard +3%",
        "adaptation_objective": "target input objective + source replay objective + 0.01 relative layer anchor",
        "adaptation_selection": "0.5 relative source validation + 0.5 relative target input; source MSE guard +10%; epoch-zero fallback",
        "target_future_labels_used_for_training_or_selection": False,
        "test_status": "previously inspected development period; no independent confirmation claimed",
        "scaling_convention": "unchanged cached per-domain training-period scalers for comparison",
        "code_sha256": {n: sha(Path(__file__).with_name(n)) for n in ("progressive_tcn.py", "progressive_train.py", "residual_tcn.py", "model.py", "data.py")},
        "data_sha256": {p.name: sha(p) for p in sorted((PROC_DIR/ds).glob("*.csv"))},
        "initial_checkpoint_sha256": {str(p.relative_to(MODEL_DIR)): sha(p) for p in init_files}}
    manifest_path = out/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Run settings/code changed; use a new experiment folder")
    write_json(manifest_path, manifest)
    for seed in args.seeds:
        seed_dir = dest/f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        def log(message):
            text = f"{ds} seed={seed} {message}"
            print(text, flush=True)
            with (out/f"seed_{seed}.log").open("a", encoding="utf-8") as f:
                f.write(text+"\n")
        initial = ProgressiveLiteTCN(cfg)
        initial.load_state_dict(torch.load(MODEL_DIR/"residual_litetcn"/ds/f"seed_{seed}"/"source_model.pt", weights_only=True))
        scores = {}
        for mode in MODES:
            record = out/f"seed_{seed}_{mode}.json"
            if record.exists() and (seed_dir/f"{mode}.pt").exists():
                scores[mode] = json.loads(record.read_text())["score"]
                continue
            model, history = source_fit(initial, src["train"], src["val"], cfg, mode, seed, args, log)
            torch.save(model.state_dict(), seed_dir/f"{mode}.pt")
            write_json(record, history)
            scores[mode] = history["score"]
        chosen = min(scores, key=scores.get)
        source = ProgressiveLiteTCN(cfg)
        source.load_state_dict(torch.load(seed_dir/f"{chosen}.pt", weights_only=True))
        torch.save(source.state_dict(), seed_dir/"source_model.pt")
        adapt_record = out/f"seed_{seed}_adaptation.json"
        if adapt_record.exists() and (seed_dir/"adapted_model.pt").exists():
            adaptation = json.loads(adapt_record.read_text())
        else:
            adapted, adaptation = adapt_fit(source, src["train"], src["val"], tgt["train"][0], tgt["val"][0], cfg, seed, args, log)
            torch.save(adapted.state_dict(), seed_dir/"adapted_model.pt")
            write_json(adapt_record, adaptation)
        selection = {"seed": seed, "source_control": chosen, "source_scores": scores,
                     "adaptation_round": adaptation["selected_round"], "adaptation_proxy": adaptation["proxy"],
                     "target_future_labels_used_for_training_or_selection": False}
        write_json(out/f"seed_{seed}_selection.json", selection)
        if seed == 0:
            save_meta(ds, data, dest/"meta.json")
            meta = json.loads((dest/"meta.json").read_text())
            meta.update(model_spec={"architecture": FAMILY}, training_seed=0, adaptation="progressive_input_only", experiment=FAMILY)
            write_json(dest/"meta.json", meta)
            write_json(dest/"selection.json", selection)
            for name in ("source_model", "adapted_model"):
                (dest/f"{name}.pt").write_bytes((seed_dir/f"{name}.pt").read_bytes())
    freeze = {"timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(), "test_evaluated_at_freeze": False,
              "checkpoint_sha256": {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob("*.pt"))}}
    freeze_path = out/"selection_frozen.json"
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text())["checkpoint_sha256"] == freeze["checkpoint_sha256"]
    else:
        write_json(freeze_path, freeze)
    write_json(out/"training_time.json", {"seconds_this_invocation": time.perf_counter()-started})


def evaluate_dataset(ds, seeds):
    data = prepare(ds)
    cfg = data["cfg"]
    out, dest = RESULTS_DIR/FAMILY/ds, MODEL_DIR/FAMILY/ds
    assert (out/"selection_frozen.json").exists()
    rows, peaks, horizons, examples = [], [], [], {}
    for seed in seeds:
        for name in (*MODES, "source_model", "adapted_model"):
            model = ProgressiveLiteTCN(cfg)
            model.load_state_dict(torch.load(dest/f"seed_{seed}"/f"{name}.pt", weights_only=True))
            for role in ("source", "target"):
                dom = data[role]
                metrics, p = evaluate(ds, cfg, model, dom, fixed_scale(ds, role, dom["scaler"]))
                rows.append({"dataset": ds, "seed": seed, "domain": role, "method": name, **metrics})
                key = cfg["columns"].index(cfg["key"])
                truth = dom["test"][1].numpy()
                sd = dom["scaler"].std[key]
                peaks.append({"dataset": ds, "seed": seed, "domain": role, "method": name,
                              **peak_metrics(truth[..., key]*sd, p[..., key]*sd)})
                for h in (1, 3, 6, 12, 24):
                    horizons.append({"dataset": ds, "domain": role, "seed": seed, "method": name, "horizon_hours": h,
                        "key_RMSE": float(np.sqrt(np.mean((p[:, :h, key]-truth[:, :h, key])**2))*sd)})
                if seed == 0:
                    ix = np.linspace(0, len(p)-1, min(256, len(p)), dtype=int)
                    examples[f"{role}_{name}"] = p[ix]
                    examples[f"{role}_X"], examples[f"{role}_Y"] = (a.numpy()[ix] for a in dom["test"])
                    examples[f"{role}_indices"] = ix
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"metrics_all.csv", index=False)
    pd.DataFrame(peaks).to_csv(out/"peak_metrics.csv", index=False)
    pd.DataFrame(horizons).to_csv(out/"horizon_metrics.csv", index=False)
    np.savez_compressed(out/"forecast_examples.npz", **examples)
    print(ds+" FROZEN TEST EVALUATION\n"+frame.query("domain=='target'").groupby("method")[["MSE", "key_RMSE"]].mean().to_string(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=("climate", "energy"), default=["climate", "energy"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--adapt-rounds", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    # Complete every dataset's selection before seeing any new test results.
    for ds in a.datasets:
        fit_dataset(ds, a)
    for ds in a.datasets:
        evaluate_dataset(ds, a.seeds)


if __name__ == "__main__":
    main()
