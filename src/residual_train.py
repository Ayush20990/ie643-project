"""Train/adapt the original LiteTCN's residual forecast head without target futures."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DATASETS, MODEL_DIR, PROC_DIR, RESULTS_DIR
from .data import batches, prepare, save_meta
from .diagnose_peaks import peak_metrics
from .improve import evaluate, fixed_scale, write_json
from .model import make_forecaster
from .refined_models import ridge_path
from .residual_tcn import ResidualLiteTCN
from .strict_models import input_pretext
from .train import predict

SOURCE_RIDGE = (.0001, .001, .01, .1, 1., 10., 100.)
ANCHORS = (.001, .01, .1, 1., 10.)
REPLAY = (.25, 1.)


@torch.no_grad()
def collect(model, x, y):
    base, design = [], []
    model.eval()
    for xb in batches(x, bs=256, shuffle=False):
        b, f = model.baseline_and_design(xb)
        base.append(b)
        design.append(f)
    b, f = torch.cat(base), torch.cat(design)
    return b, f, y


def statistics(cached):
    baseline, design, y = cached
    features = design.permute(1, 0, 2).double()
    residual = (y-baseline).permute(2, 0, 1).double()
    return features.transpose(1, 2)@features/len(y), features.transpose(1, 2)@residual/len(y)


def channel_error(cached, weight):
    base, design, truth = cached
    prediction = base+torch.einsum("bcf,cfh->bhc", design, weight)
    return (prediction-truth).square().mean((0, 1)).numpy()


def run_dataset(ds, seeds):
    data = prepare(ds)
    cfg, src, tgt = data["cfg"], data["source"], data["target"]
    out, dest = RESULTS_DIR/"residual_litetcn"/ds, MODEL_DIR/"residual_litetcn"/ds
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": ds, "seeds": seeds, "source_ridge": SOURCE_RIDGE, "anchors": ANCHORS, "source_replay": REPLAY,
        "code_sha256": {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                        for n in ("residual_tcn.py", "residual_train.py", "model.py", "data.py")},
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((PROC_DIR/ds).glob("*.csv"))},
        "protocol": "source-trained original LiteTCN encoder; DLinear initialized shared skip; source residual-head ridge; within-input target head adaptation with source replay/anchor and zero-residual fallback",
        "test_status": "reused development period; not independent confirmation"}
    write_json(out/"manifest.json", manifest)
    frozen = {}
    for seed in seeds:
        print(f"{ds} s{seed}: fit original LiteTCN residual head", flush=True)
        existing = MODEL_DIR/"improved"/ds/f"seed_{seed}"
        tcn, linear = make_forecaster(cfg), make_forecaster(cfg, {"architecture": "dlinear"})
        tcn.load_state_dict(torch.load(existing/"tcn3.pt", weights_only=True))
        linear.load_state_dict(torch.load(existing/"dlinear.pt", weights_only=True))
        model = ResidualLiteTCN(cfg).initialize(tcn, linear)
        source_train, source_val = collect(model, *src["train"]), collect(model, *src["val"])
        gt, gv = input_pretext(tgt["train"][0], cfg["H"]), input_pretext(tgt["val"][0], cfg["H"])
        target_train, target_val = collect(model, *gt), collect(model, *gv)
        gs, rs = statistics(source_train)
        zero = torch.zeros_like(model.head_weights())
        source_weight = zero.clone()
        best_source = channel_error(source_val, zero)
        source_choice = [{"channel": c, "ridge": None, "MSE": float(v)} for c, v in enumerate(best_source)]
        search = []
        for penalty, weight in ridge_path(gs, rs, SOURCE_RIDGE):
            error = channel_error(source_val, weight)
            for c in range(model.C):
                search.append({"channel": c, "ridge": penalty, "MSE": float(error[c])})
                if error[c] < best_source[c]:
                    best_source[c] = error[c]
                    source_weight[c] = weight[c]
                    source_choice[c] = search[-1]
        model.set_head(source_weight)
        source = copy.deepcopy(model)
        base_s, base_t = channel_error(source_val, zero), channel_error(target_val, zero)
        best_proxy = np.ones(model.C)
        adapted_weight = zero.clone()
        adapted_choice = [{"channel": c, "method": "zero_residual", "proxy": 1.} for c in range(model.C)]
        for name, weight in (("source_head", source_weight),):
            sv, tv = channel_error(source_val, weight), channel_error(target_val, weight)
            score = .5*sv/np.maximum(base_s, 1e-8)+.5*tv/np.maximum(base_t, 1e-8)
            for c in range(model.C):
                if sv[c] <= 1.1*base_s[c] and score[c] < best_proxy[c]:
                    best_proxy[c], adapted_weight[c] = score[c], weight[c]
                    adapted_choice[c] = {"channel": c, "method": name, "proxy": float(score[c])}
        gg, rr = statistics(target_train)
        adaptation_search = []
        for replay in REPLAY:
            for anchor, weight in ridge_path(gg+replay*gs, rr+replay*rs, ANCHORS, source_weight):
                sv, tv = channel_error(source_val, weight), channel_error(target_val, weight)
                score = .5*sv/np.maximum(base_s, 1e-8)+.5*tv/np.maximum(base_t, 1e-8)
                for c in range(model.C):
                    eligible = bool(sv[c] <= 1.1*base_s[c])
                    row = {"channel": c, "method": "input_residual", "source_replay": replay, "anchor": anchor,
                           "source_val_MSE": float(sv[c]), "input_pretext_MSE": float(tv[c]),
                           "proxy": float(score[c]), "eligible": eligible}
                    adaptation_search.append(row)
                    if eligible and score[c] < best_proxy[c]:
                        best_proxy[c], adapted_weight[c] = score[c], weight[c]
                        adapted_choice[c] = row
        model.set_head(adapted_weight)
        plain = copy.deepcopy(source)
        plain.set_head(zero)
        seed_dir = dest/f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        for name, m in (("source", source), ("adapted", model)):
            torch.save(m.state_dict(), seed_dir/f"{name}_model.pt")
        pd.DataFrame(search).to_csv(out/f"seed_{seed}_source_search.csv", index=False)
        pd.DataFrame(adaptation_search).to_csv(out/f"seed_{seed}_adaptation_search.csv", index=False)
        choice = {"seed": seed, "source_head": source_choice, "adapted_head": adapted_choice,
                  "zero_residual_baseline": "DLinear source forecast, exactly merged into the shared temporal skip",
                  "target_forecasting_labels_used": False, "source_guard_reference": "zero-residual validation MSE per channel"}
        write_json(out/f"seed_{seed}_selection.json", choice)
        frozen[seed] = {"linear_skip_only": plain.eval(), "source_residual_tcn": source.eval(), "adapted_residual_tcn": model.eval()}
        if seed == seeds[0]:
            for name, m in (("source", source), ("adapted", model)):
                torch.save(m.state_dict(), dest/f"{name}_model.pt")
            save_meta(ds, data, dest/"meta.json")
            meta = json.loads((dest/"meta.json").read_text())
            meta.update(model_spec={"architecture": "residual_litetcn"}, training_seed=seed,
                        adaptation="anchored_input_residual", experiment="residual_litetcn")
            write_json(dest/"meta.json", meta)
            write_json(dest/"selection.json", choice)
    write_json(out/"selection_frozen.json", {"test_evaluated_at_freeze": False,
               "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    rows, peaks, examples = [], [], {}
    for seed, models in frozen.items():
        for role in ("source", "target"):
            dom = data[role]
            indices = np.linspace(0, len(dom["test"][0])-1, min(256, len(dom["test"][0])), dtype=int)
            key = cfg["columns"].index(cfg["key"])
            for name, m in models.items():
                metrics, p = evaluate(ds, cfg, m, dom, fixed_scale(ds, role, dom["scaler"]))
                rows.append({"dataset": ds, "seed": seed, "domain": role, "method": name, **metrics})
                sd = dom["scaler"].std[key]
                peaks.append({"dataset": ds, "seed": seed, "domain": role, "method": name,
                              **peak_metrics(dom["test"][1].numpy()[..., key]*sd, p[..., key]*sd)})
                if seed == seeds[0]:
                    examples[f"{role}_{name}"] = p[indices]
            if seed == seeds[0]:
                examples[f"{role}_X"], examples[f"{role}_Y"] = (v.numpy()[indices] for v in dom["test"])
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
    torch.set_num_threads(4)
    for ds in args.datasets:
        run_dataset(ds, args.seeds)


if __name__ == "__main__":
    main()
