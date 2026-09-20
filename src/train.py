"""Source training, unsupervised adaptation (ours + baselines) and supervised reference models."""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from .data import batches, infinite
from .losses import DomainDiscriminator, coral, l2_sp, mmd_rbf, spectral_align
from .model import LiteTCNForecaster

torch.set_num_threads(max(1, torch.get_num_threads()))

# Unsupervised adaptation recipes. Every recipe starts from the SAME source checkpoint and only ever
# sees target *inputs* (no target future values).  Weights: lam_* for each term of the loss.
METHODS = {
    "dann":       dict(dann=0.1),
    "coral":      dict(coral=1.0),
    "mmd":        dict(mmd=1.0),
    "care":       dict(mmd=1.0, coral=1.0, ssl=1.0, anchor=1e-2),   # CARE-DA v1 (as designed)
    "care_noalign":  dict(ssl=1.0, anchor=1e-2),                     # ablations
    "care_nossl":    dict(mmd=1.0, coral=1.0, anchor=1e-2),
    "care_noanchor": dict(mmd=1.0, coral=1.0, ssl=1.0),
    # v2: drop input reconstruction; add forecast-consistent pretext + spectral align
    "care_v2":    dict(mmd=1.0, mmd_h=0.5, inner=0.5, consist=0.3, spectral=0.3, anchor=1e-2),
}


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


@torch.no_grad()
def predict(model, X, bs=1024):
    model.eval()
    return torch.cat([model(x) for x in batches(X, bs=bs, shuffle=False)]).numpy()


@torch.no_grad()
def embed(model, X, bs=1024):
    model.eval()
    return torch.cat([model(x, return_z=True)[1] for x in batches(X, bs=bs, shuffle=False)])


def eval_mse(model, X, Y):
    return float(np.mean((predict(model, X) - Y.numpy()) ** 2))


def build_model(cfg):
    return LiteTCNForecaster(cfg["L"], cfg["H"], len(cfg["columns"]))


def train_supervised(model, train, val, epochs=20, lr=1e-3, ssl=0.5, bs=256, seed=0, log=None, tag="source"):
    """Supervised training with early model selection on a validation split of the SAME domain."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, hist = math.inf, None, []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in batches(*train, bs=bs, gen=gen):
            loss = F.mse_loss(model(xb), yb)
            if ssl:
                loss = loss + ssl * model.masked_recon_loss(xb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item() * len(xb), n + len(xb)
        sched.step()
        v = eval_mse(model, *val)
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "val_mse": v})
        if log:
            log(f"[{tag}] ep {ep + 1:02d} train {tot / n:.4f} val_mse {v:.4f}")
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, hist


def adapt(src_model, src, tgt, recipe, epochs=15, lr=5e-4, bs=256, seed=0, log=None, tag="care",
          monitor=None, select="last"):
    """Unsupervised DA.  L = MSE_src + a*MMD + b*CORAL + c*DANN + d*SSL_tgt + e*L2SP
    + inner-forecast / consistency / spectral terms (CARE-DA v2).
    `monitor` = (X_tgt_test, Y_tgt_test) is used ONLY to draw adaptation curves, never for selection.
    `select`: 'last' (fixed budget) or 'src_mmd' (min source-val + feature MMD, no target labels)."""
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = copy.deepcopy(src_model)
    ref = {n: p.detach().clone() for n, p in src_model.named_parameters()}
    disc = DomainDiscriminator(model.head.in_features) if recipe.get("dann") else None
    params = list(model.parameters()) + (list(disc.parameters()) if disc else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    Xs, Ys = src["train"]
    Xt = tgt["train"][0]  # target INPUT windows only
    steps = max(1, math.ceil(len(Xt) / bs))
    it_s, it_t = infinite(Xs, Ys, bs, gen), infinite(Xt, None, bs, gen)
    probe_s = src["val"][0][:1024]
    probe_t = tgt["val"][0][:1024]
    hist = []
    need_h = bool(recipe.get("mmd_h"))
    best_score, best_state = math.inf, None

    def snapshot(ep, comps):
        zs, zt = embed(model, probe_s), embed(model, probe_t)
        row = {"epoch": ep, **comps, "mmd_feat": float(mmd_rbf(zs, zt)),
               "src_val_mse": eval_mse(model, *src["val"])}
        if monitor is not None:
            row["tgt_test_mse"] = eval_mse(model, *monitor)
        hist.append(row)
        if log:
            log(f"[{tag}] ep {ep:02d} " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"))
        return row

    snapshot(0, {})
    total_steps = epochs * steps
    for ep in range(1, epochs + 1):
        model.train()
        acc = {}
        for step in range(steps):
            xs, ys = next(it_s)
            xt = next(it_t)
            if need_h:
                ps, zs, hs = model(xs, return_h=True)
                _, zt, ht = model(xt, return_h=True)
            else:
                ps, zs = model(xs, return_z=True)
                _, zt = model(xt, return_z=True)
            terms = {"sup": F.mse_loss(ps, ys)}
            if recipe.get("mmd"):
                terms["mmd"] = recipe["mmd"] * mmd_rbf(zs, zt)
            if recipe.get("mmd_h"):
                terms["mmd_h"] = recipe["mmd_h"] * mmd_rbf(hs, ht)
            if recipe.get("coral"):
                terms["coral"] = recipe["coral"] * coral(zs, zt)
            if recipe.get("dann"):
                p = ((ep - 1) * steps + step) / total_steps
                lam = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
                logits = disc(torch.cat([zs, zt]), lam)
                lbl = torch.cat([torch.ones(len(zs)), torch.zeros(len(zt))])
                terms["dann"] = recipe["dann"] * F.binary_cross_entropy_with_logits(logits, lbl)
            if recipe.get("ssl"):
                terms["ssl"] = recipe["ssl"] * model.masked_recon_loss(xt)
            if recipe.get("inner"):
                terms["inner"] = recipe["inner"] * model.inner_forecast_loss(xt)
            if recipe.get("consist"):
                terms["consist"] = recipe["consist"] * model.forecast_consistency_loss(xt)
            if recipe.get("spectral"):
                terms["spectral"] = recipe["spectral"] * spectral_align(xs, xt)
            if recipe.get("anchor"):
                terms["anchor"] = recipe["anchor"] * l2_sp(model, ref)
            loss = sum(terms.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            for k, v in terms.items():
                acc[k] = acc.get(k, 0.0) + v.item() / steps
        acc["total"] = sum(acc.values())
        row = snapshot(ep, acc)
        score = row["src_val_mse"] + row["mmd_feat"]
        if score < best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    if select == "src_mmd" and best_state is not None:
        model.load_state_dict(best_state)
    return model, hist
