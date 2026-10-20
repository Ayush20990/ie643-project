"""Source-only small-subset fitting sanity check; never used for deployment."""
import copy

import numpy as np
import pandas as pd
import torch

from .config import MODEL_DIR, RESULTS_DIR
from .diagnose_peaks import peak_metrics
from .model import make_forecaster
from .peak_data import prepare_peak
from .peak_model import PeakLiteTCN, forecast_loss
from .train import set_seed


def main():
    torch.set_num_threads(2)
    rows = []
    for ds in ("climate", "energy"):
        data = prepare_peak(ds)
        cfg, src = data["cfg"], data["source"]
        set_seed(123)
        indices = torch.randperm(len(src["train"][0]))[:64]
        key = cfg["columns"].index(cfg["key"])
        old = make_forecaster(cfg, {"architecture": "patch_mlp"})
        old.load_state_dict(torch.load(MODEL_DIR/"improved"/ds/"seed_0"/"source_model.pt", weights_only=True))
        for name, model, train in (("previous_patch_mlp", old, src["train"]),
                                    ("new_litetcn", PeakLiteTCN(cfg), src["train_packed"])):
            x, y = (a[indices] for a in train)
            # Disable dropout deliberately: this probes capacity to fit a small
            # source sample, not generalization, hyperparameters or test quality.
            model.eval()
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
            for step in range(129):
                if step in (0, 32, 64, 128):
                    with torch.no_grad():
                        p = model(x).numpy()
                    sd, mu = src["scaler"].std[key], src["scaler"].mean[key]
                    rows.append({"dataset": ds, "model": name, "updates": step, "source_samples": 64,
                        **peak_metrics(y.numpy()[..., key]*sd+mu, p[..., key]*sd+mu)})
                if step == 128:
                    break
                loss = forecast_loss(model(x), y, key, shape=True)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR/"peak_diagnosis"/"source_overfit_check.csv", index=False)
    print(frame[["dataset", "model", "updates", "RMSE", "mean_actual_range", "mean_predicted_range"]].to_string(index=False))


if __name__ == "__main__":
    main()
