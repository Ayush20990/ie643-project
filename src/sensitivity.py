"""Post-hoc sensitivity of CARE-DA to the SSL weight (lambda_R) and the anchor weight (lambda_A).

Usage:  python -m src.sensitivity --dataset climate
Starts from the saved seed-0 source model. Reported for transparency - these numbers are NOT used to
choose the hyper-parameters of the main results table.
"""
import argparse
import json

import numpy as np
import torch

from .config import MODEL_DIR, RESULTS_DIR
from .data import prepare
from .run_experiments import ADAPT_EPOCHS, ADAPT_LR
from .train import adapt, build_model, eval_mse

GRID = {
    "lambda_R": [0.0, 0.1, 0.3, 1.0],
    "lambda_A": [0.0, 1e-3, 1e-2, 1e-1],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ds = ap.parse_args().dataset
    torch.set_num_threads(4)
    data = prepare(ds)
    S, T = data["source"], data["target"]
    src = build_model(data["cfg"])
    src.load_state_dict(torch.load(MODEL_DIR / ds / "source_model.pt"))
    base = {"mmd": 1.0, "coral": 1.0, "ssl": 1.0, "anchor": 1e-2}
    out = {"source_only": {"tgt": eval_mse(src, *T["test"]), "src": eval_mse(src, *S["test"])}}
    for name, key in [("lambda_R", "ssl"), ("lambda_A", "anchor")]:
        out[name] = []
        for v in GRID[name]:
            recipe = {**base, key: v}
            m, _ = adapt(src, S, T, recipe, ADAPT_EPOCHS, ADAPT_LR, seed=0)
            row = {"value": v, "tgt_test_mse": eval_mse(m, *T["test"]), "src_test_mse": eval_mse(m, *S["test"])}
            out[name].append(row)
            print(ds, name, row, flush=True)
    (RESULTS_DIR / ds / "sensitivity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
