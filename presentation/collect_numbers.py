"""Collect dataset statistics and experiment results into numbers.json for the deck/report builders."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATASETS, RESULTS_DIR  # noqa: E402
from src.data import load_domain  # noqa: E402

out = {"datasets": {}, "results": {}}
for ds, cfg in DATASETS.items():
    info = {}
    for role in ["source", "target"]:
        df = load_domain(ds, cfg[role])
        info[role] = {"rows": int(len(df)), "start": str(df.date.min().date()), "end": str(df.date.max().date()),
                      "series": int(df.series.nunique()) if "series" in df else 1,
                      "mean": {c: float(df[c].mean()) for c in cfg["columns"]},
                      "std": {c: float(df[c].std()) for c in cfg["columns"]}}
    info.update({k: cfg[k] for k in ["L", "H", "columns", "key", "source_label", "target_label", "title"]})
    out["datasets"][ds] = info
    p = RESULTS_DIR / ds / "summary_target.csv"
    if p.exists():
        s = pd.read_csv(p, index_col=0)
        out["results"][ds] = {m: {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
                              for m, row in s.iterrows()}
        so, orc = s.loc["source_only", "MSE_mean"], s.loc["oracle_finetune", "MSE_mean"]
        for m in s.index:
            out["results"][ds][m]["rel_change"] = float(s.loc[m, "MSE_mean"] / so - 1)
            out["results"][ds][m]["gcr_means"] = float((so - s.loc[m, "MSE_mean"]) / (so - orc))
        allm = pd.read_csv(RESULTS_DIR / ds / "metrics_all.csv")
        out["results"][ds]["_n_seeds"] = int(allm[allm.seed >= 0].seed.nunique())
        src = allm[(allm.eval_domain == "source") & (allm.method == "source_only")]
        out["results"][ds]["_source_test_mse"] = float(src.MSE.mean())
    sp = RESULTS_DIR / ds / "sensitivity.json"
    if sp.exists():
        out.setdefault("sensitivity", {})[ds] = json.loads(sp.read_text())
    v2p = RESULTS_DIR / ds / "care_v2_compare.csv"
    if v2p.exists():
        v2 = pd.read_csv(v2p, index_col=0)
        if "care_v2" in v2.index:
            row = v2.loc["care_v2"]
            so = float(v2.loc["source_only", "MSE"]) if "source_only" in v2.index else float(row["MSE"])
            out["results"].setdefault(ds, {})["care_v2"] = {
                "MSE_mean": float(row["MSE"]), "MSE_std": None,
                "MAE_mean": float(row["MAE"]), "MASE_mean": float(row["MASE"]),
                "key_RMSE_mean": float(row["key_RMSE"]), "key_MAE_mean": float(row["key_MAE"]),
                "rel_change": float(row["rel_change_%"]) / 100.0,
                "gcr_means": None, "_seeds": 1,
            }
log = RESULTS_DIR / "climate" / "train_log.txt"
if log.exists():
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.startswith("LiteTCN params"):
            out["params_climate"] = line.split(":")[1].strip()
(Path(__file__).parent / "numbers.json").write_text(json.dumps(out, indent=1))
print(json.dumps({ds: list(v.keys())[:3] for ds, v in out["results"].items()}, indent=0))
