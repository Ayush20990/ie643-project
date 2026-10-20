"""Exploratory compact-model adaptation and architecture selection using observed target history.

Source validation alone need not rank models by transfer quality. Compare the
first sweep's source/adapter with DLinear source/adapters using a common,
input-only validation proxy. Reuses source checkpoints; never reads target test
futures until the selection record and deployment checkpoints have been saved.
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, RESULTS_DIR
from .data import prepare, save_meta
from .improve import RECIPES, adapt_history, evaluate, fixed_scale, inner_score, validation_mse, write_json
from .inference import load_bundle
from .model import make_forecaster


def run(ds, args):
    base = RESULTS_DIR / "improved" / ds
    if not (base / "manifest.json").exists():
        print(f"{ds}: no source sweep yet", flush=True)
        return
    data = prepare(ds)
    cfg, src, tgt = data["cfg"], data["source"], data["target"]
    out, dest = RESULTS_DIR / "transfer_selection" / ds, MODEL_DIR / "transfer_selection" / ds
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": ds, "seeds": args.seeds, "adapt_rounds": args.adapt_epochs,
                "recipes": RECIPES, "code_sha256": hashlib.sha256(__file_bytes()).hexdigest(),
                "source_manifest": json.loads((base / "manifest.json").read_text()),
                "selection": "mean(source_val / legacy_source_val, target_input_pretext / legacy_input_pretext); source guard 1.1 * legacy_source_val",
                "status": "exploratory extension after viewing the initial development test results"}
    p = out / "manifest.json"
    if p.exists() and json.loads(p.read_text()) != manifest:
        raise ValueError(f"Settings differ from {p}")
    write_json(p, manifest)
    probe = tgt["val"][0][::max(1, len(tgt["val"][0]) // 512)][:512]
    denominators = {r: fixed_scale(ds, r, data[r]["scaler"]) for r in ("source", "target")}
    for seed in args.seeds:
        if (out / f"seed_{seed}.csv").exists():
            print(f"{ds} seed {seed}: complete", flush=True)
            continue
        if not (base / f"seed_{seed}.csv").exists():
            print(f"{ds} seed {seed}: source sweep pending", flush=True)
            continue
        print(f"{ds} seed {seed}: adapting DLinear", flush=True)
        saved = MODEL_DIR / "improved" / ds / f"seed_{seed}"
        seed_dest = dest / f"seed_{seed}"
        seed_dest.mkdir(exist_ok=True)
        selection = json.loads((base / f"seed_{seed}_selection.json").read_text())
        source = make_forecaster(cfg, {"architecture": "dlinear"})
        source.load_state_dict(torch.load(saved / "dlinear.pt", weights_only=True))
        main_source = make_forecaster(cfg, selection["model_spec"])
        main_source.load_state_dict(torch.load(saved / "source_model.pt", weights_only=True))
        candidates = {"main_source": main_source, "dlinear_source": source}
        specs = {"main_source": selection["model_spec"], "dlinear_source": {"architecture": "dlinear"}}
        for name, recipe in RECIPES.items():
            main = make_forecaster(cfg, selection["model_spec"])
            main.load_state_dict(torch.load(saved / f"{name}.pt", weights_only=True))
            candidates[f"main_{name}"], specs[f"main_{name}"] = main, selection["model_spec"]
            model, hist, _, _ = adapt_history(source, src, tgt["train"][0], probe, recipe,
                args.adapt_epochs, seed, lambda msg: print(f"{ds} s{seed} DLinear {name} {msg}", flush=True))
            candidates[f"dlinear_{name}"], specs[f"dlinear_{name}"] = model, {"architecture": "dlinear"}
            write_json(out / f"seed_{seed}_{name}.json", hist)
            torch.save(model.state_dict(), seed_dest / f"{name}.pt")
        legacy_meta, legacy_models = load_bundle(ds)
        legacy = legacy_models["source_model"]
        ref_s, ref_t = validation_mse(legacy, src), inner_score(legacy, probe)
        scores = {}
        for name, model in candidates.items():
            sv, tv = validation_mse(model, src), inner_score(model, probe)
            scores[name] = {"source_val_mse": sv, "target_input_mse": tv,
                            "proxy": .5 * sv / ref_s + .5 * tv / ref_t,
                            "eligible": sv <= 1.1 * ref_s}
        eligible = [name for name, score in scores.items() if score["eligible"]]
        # A common reference defines the proxy. If every candidate fails the
        # source guard, preserve the original source checkpoint explicitly.
        if eligible:
            chosen = min(eligible, key=lambda name: scores[name]["proxy"])
            chosen_model, spec = candidates[chosen], specs[chosen]
            chosen_source = source if chosen.startswith("dlinear") else main_source
            recipe = next((r for r in RECIPES if chosen.endswith(r)), "source_only")
        else:
            chosen, chosen_model, chosen_source, spec, recipe = "legacy_source", legacy, legacy, {"architecture": "litetcn"}, "source_only"
        record = {"seed": seed, "chosen": chosen, "model_spec": spec, "adaptation": recipe,
                  "source_reference_mse": ref_s, "input_reference_mse": ref_t, "scores": scores,
                  "future_target_values_used_for_selection": False,
                  "note": "Exploratory model family selection on the reused development benchmark"}
        write_json(out / f"seed_{seed}_selection.json", record)
        # Freeze all selection artifacts BEFORE evaluating test futures.
        torch.save(chosen_source.state_dict(), seed_dest / "source_model.pt")
        torch.save(chosen_model.state_dict(), seed_dest / "adapted_model.pt")
        if seed == args.seeds[0]:
            torch.save(chosen_source.state_dict(), dest / "source_model.pt")
            torch.save(chosen_model.state_dict(), dest / "adapted_model.pt")
            save_meta(ds, data, dest / "meta.json")
            meta = json.loads((dest / "meta.json").read_text())
            meta.update(model_spec=spec, adaptation=recipe, training_seed=seed, experiment="transfer_selection")
            write_json(dest / "meta.json", meta)
            write_json(dest / "selection.json", record)
        rows, examples = [], {}
        evaluate_models = {**candidates, "history_selected_source": chosen_source,
                           "history_selected_adapted": chosen_model,
                           "legacy_care_v2_seed0": legacy_models["adapted_model"]}
        for role in ("source", "target"):
            for name, model in evaluate_models.items():
                metric, pred = evaluate(ds, cfg, model, data[role], denominators[role])
                rows.append({"dataset": ds, "seed": seed, "method": name, "domain": role, **metric})
                if seed == args.seeds[0] and name in ("history_selected_source", "history_selected_adapted", "legacy_care_v2_seed0"):
                    indices = np.linspace(0, len(pred) - 1, min(256, len(pred)), dtype=int)
                    examples[f"{role}_{name}"] = pred[indices]
            if seed == args.seeds[0]:
                examples[f"{role}_X"] = data[role]["test"][0][indices].numpy()
                examples[f"{role}_Y"] = data[role]["test"][1][indices].numpy()
                examples[f"{role}_indices"] = indices
        if examples:
            np.savez_compressed(out / "forecast_examples.npz", **examples)
        pd.DataFrame(rows).to_csv(out / f"seed_{seed}.csv", index=False)
        print(f"{ds} seed {seed}: selected {chosen} using validation only", flush=True)
    completed = sorted(out.glob("seed_[0-9]*.csv"))
    if completed:
        df = pd.concat([pd.read_csv(p) for p in completed], ignore_index=True)
        df.to_csv(out / "metrics_all.csv", index=False)
        summary = df.groupby(["domain", "method"])[["MSE", "MAE", "MASE_train", "key_RMSE"]].agg(["mean", "std", "count"])
        summary.columns = ["_".join(c) for c in summary.columns]
        summary.to_csv(out / "summary.csv")


def __file_bytes():
    from pathlib import Path
    return Path(__file__).read_bytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--adapt-epochs", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(2)
    for ds in args.datasets:
        run(ds, args)


if __name__ == "__main__":
    main()
