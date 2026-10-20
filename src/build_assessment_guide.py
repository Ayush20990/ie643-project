"""Render the assessment study guide with tables taken from saved experiments.

This is documentation only: it does not fit models or change model selection.
Requires the optional Python-Markdown package for the portable HTML export.
"""
import base64
import html
import json
from pathlib import Path

import markdown
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image

from .config import ROOT, RESULTS_DIR
from .strict_report import previous


REPORT = ROOT / "report"


def md_table(rows, columns=None):
    frame = pd.DataFrame(rows, columns=columns)
    def value(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("|", "/")
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
        *("| " + " | ".join(map(value, row)) + " |" for row in frame.itertuples(index=False, name=None)),
    ])


def metrics(family, ds):
    frame = pd.read_csv(RESULTS_DIR / family / ds / "metrics_all.csv")
    return frame[frame.domain == "target"]


def result_row(label, frame):
    return {"Model / policy": label, "Standardized MSE": frame.MSE.mean(),
            "Key-variable RMSE": frame.key_RMSE.mean()}


def build_tables():
    tables = {}
    rows = []
    for ds in ("climate", "energy", "finance"):
        frame = pd.read_csv(RESULTS_DIR / ds / "metrics_all.csv")
        frame = frame[frame.eval_domain == "target"]
        for method in ("source_only", "care", "mmd", "coral", "dann"):
            d = frame[frame.method == method]
            rows.append({"Dataset": ds, "Original method (3 seeds)": method, "MSE": d.MSE.mean()})
    tables["ORIGINAL_TABLE"] = md_table(rows)

    rows = []
    for ds in ("climate", "energy", "finance"):
        frame = pd.read_csv(RESULTS_DIR / ds / "care_v2_compare.csv")
        for _, r in frame.iterrows():
            rows.append({"Dataset": ds, "Saved seed-0 method": r.iloc[0], "MSE": r.MSE, "Key RMSE": r.key_RMSE})
    tables["V2_TABLE"] = md_table(rows)

    rows = []
    for ds in ("climate", "energy", "finance"):
        f = metrics("improved", ds)
        for method in ("tcn3", "tcn5", "tcn7", "patch_mlp", "dlinear"):
            d = f[f.method == method]
            rows.append({"Dataset": ds, "Source-trained candidate": method, "Target MSE": d.MSE.mean(), "Target key RMSE": d.key_RMSE.mean()})
    tables["DEPTH_TABLE"] = md_table(rows)

    for ds in ("climate", "energy", "finance"):
        initial = metrics("improved", ds)
        strict = metrics("strict_refined", ds)
        context = metrics("strict_context", ds)
        rows = [result_row("Original CARE-DA v2, fixed seed 0", initial[initial.method == "legacy_care_v2_seed0"]),
                result_row("First validation-selected adapted model", initial[initial.method == "selected_adapted"])]
        if ds == "energy":
            rows.extend([result_row("DLinear source-only (all seeds)", initial[initial.method == "dlinear"]),
                         result_row("Guarded DLinear family policy", previous(ds))])
        rows.extend([result_row("Strict input-only ensemble", strict[strict.method == "strict_adapted"]),
                     result_row("Shorter-context strict ensemble", context[context.method == "strict_adapted"])])
        if ds != "finance":
            for family, method, label in (
                ("peak_litetcn", "selected_adapted", "LiteTCN v3 adapted"),
                ("peak_calibrated", "local_adapted", "LiteTCN v3 + observed residual calibration"),
                ("residual_litetcn", "adapted_residual_tcn", "LiteTCN-R adapted"),
                ("progressive_litetcn", "adapted_model", "Gradual-unfreezing follow-up"),
                ("coupled_transfer", "adapted_model", "Retained TCN + predicted-driver decoder"),
                ("event_boost", "adapted_model", "Retained TCN + geometry-aware boosting"),
            ):
                f = metrics(family, ds)
                rows.append(result_row(label, f[f.method == method]))
        for method in ("seasonal_naive", "persistence", "input_mean"):
            rows.append(result_row(method, strict[strict.method == method]))
        tables[f"LONG_{ds.upper()}"] = md_table(rows)

    headline = pd.read_csv(RESULTS_DIR / "forecast_horizons" / "next_hour_headline.csv")
    rows = []
    for _, r in headline.iterrows():
        rows.append({"Dataset": r.dataset, "Source-only RMSE (degC)": r.source_key_RMSE,
                     "Adapted RMSE (degC)": r.adapted_key_RMSE, "Persistence RMSE (degC)": r.persistence_key_RMSE,
                     "Adaptation reduction (%)": r.key_RMSE_gain_from_adaptation_pct,
                     "Reduction vs persistence (%)": r.key_RMSE_gain_vs_persistence_pct})
    tables["ROLLING_HEADLINE"] = md_table(rows)

    rows, peakrows = [], []
    for ds in ("climate", "energy"):
        path = RESULTS_DIR / "forecast_horizons" / f"{ds}_next_hour_comparison.csv"
        f = pd.read_csv(path); f = f[(f.domain == "target") & (f.method == "adapted_model")]
        for _, r in f.iterrows():
            rows.append({"Dataset": ds, "Same 1-hour task": r.family, "MSE": r.MSE, "Temperature RMSE (degC)": r.key_RMSE})
        p = pd.read_csv(RESULTS_DIR / "forecast_horizons" / f"{ds}_next_hour_peak_comparison.csv")
        for _, r in p.iterrows():
            if r.method not in ("adapted_model", "persistence"):
                continue
            peakrows.append({"Dataset": ds, "Model": r.family if r.method != "persistence" else "Persistence",
                             "Peak MAE (degC)": r.MAE_at_actual_peak, "Trough MAE (degC)": r.MAE_at_actual_trough,
                             "Peak timing error (hours)": r.mean_absolute_peak_timing_steps})
    tables["ROLLING_COMPARISON"] = md_table(rows)
    tables["PEAK_TABLE"] = md_table(peakrows)

    rows = []
    for ds in ("climate", "energy"):
        f = pd.read_csv(RESULTS_DIR / "event_boost" / ds / "boost_adapter_ablation.csv")
        rows.append({"Dataset": ds, "Added-tree control": "New target trees removed", "Temperature RMSE": f[f.domain == "target"].key_RMSE.mean()})
        f = metrics("event_boost", ds)
        rows.append({"Dataset": ds, "Added-tree control": "New target-input trees retained", "Temperature RMSE": f[f.method == "adapted_model"].key_RMSE.mean()})
    tables["BOOST_ABLATION"] = md_table(rows)

    f = pd.read_csv(RESULTS_DIR / "forecast_horizons" / "efficiency.csv")
    tables["EFFICIENCY"] = md_table([
        {"Dataset": r.dataset, "Family": r.family, "Neural parameters": r.neural_parameters,
         "Stored tensor values": r.stored_tensor_values, "Deployed KB": r.adapted_forecaster_KB,
         "CPU ms / single forecast": r.batch_1_CPU_ms} for _, r in f.iterrows()
    ])

    # Keep the two domains visible, rather than reporting target results alone.
    rows = []
    for ds in ("climate", "energy"):
        f = pd.read_csv(RESULTS_DIR / "peak_next_hour" / ds / "metrics_all.csv")
        for (domain, method), g in f.groupby(["domain", "method"]):
            rows.append({"Dataset": ds, "Evaluated domain": domain, "Model": method,
                         "MSE": g.MSE.mean(), "Temperature RMSE": g.key_RMSE.mean()})
    tables["SOURCE_RETENTION"] = md_table(rows)
    return tables


def main():
    source = (REPORT / "assessment_guide.template.md").read_text(encoding="utf-8")
    for key, table in build_tables().items():
        token = "{{" + key + "}}"
        assert token in source, key
        source = source.replace(token, table)
    assert "{{" not in source, "Unresolved documentation placeholder"
    md_path = REPORT / "IE643_ASSESSMENT_GUIDE.md"
    md_path.write_text(source, encoding="utf-8")

    body = markdown.markdown(source, extensions=["tables", "fenced_code", "toc", "attr_list"],
                             extension_configs={"toc": {"toc_depth": "2-2"}})
    soup = BeautifulSoup(body, "html.parser")
    for img in soup.find_all("img"):
        p = (REPORT / img["src"]).resolve()
        assert p.is_file(), p
        with Image.open(p) as raster:
            img["width"],img["height"] = raster.size
        img["src"] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
        img["loading"] = "lazy"
    for table in soup.find_all("table"):
        wrapper = soup.new_tag("div", attrs={"class": "table-wrap", "tabindex": "0"})
        table.wrap(wrapper)
    toc = soup.find("div", class_="toc")
    nav = str(toc.extract()) if toc else ""
    css = """
    :root{--ink:#192a38;--blue:#175e8a;--paper:#fff;--line:#d9e2e9}
    *{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
    body{margin:0;background:#f1f5f8;color:var(--ink);font:16px/1.72 system-ui,-apple-system,Segoe UI,sans-serif;overflow-wrap:anywhere}
    .layout{display:grid;grid-template-columns:270px minmax(0,1060px);max-width:1390px;margin:auto}
    aside{position:sticky;top:0;height:100vh;overflow:auto;padding:30px 20px;font-size:13px}
    aside a{display:block;padding:3px 0}aside ul{padding-left:15px}aside li{padding:2px 0}
    main{background:var(--paper);padding:44px 48px 70px;min-width:0;border-left:1px solid var(--line)}
    h1{font-size:36px;line-height:1.17;letter-spacing:-1px}h2{font-size:26px;line-height:1.3;margin-top:58px;padding-top:15px;border-top:2px solid var(--line)}
    h3{font-size:20px;line-height:1.4;margin-top:30px}h4{font-size:17px;margin-bottom:6px}
    a{color:var(--blue);text-decoration-thickness:1px;text-underline-offset:3px}p{margin:14px 0}
    img{display:block;max-width:100%;height:auto;margin:22px auto;border:1px solid var(--line);border-radius:8px}
    blockquote{margin:22px 0;padding:8px 20px;border-left:4px solid #2d86b7;background:#eef6fb}
    pre{background:#132b3a;color:#f5f8fa;padding:20px;border-radius:8px;overflow:auto;font-size:13px;line-height:1.6;overflow-wrap:normal}
    code{font-size:.88em;background:#eef2f5;padding:2px 4px;border-radius:3px}pre code{background:none;padding:0;color:inherit}
    .table-wrap{overflow-x:auto;margin:20px 0;border:1px solid var(--line);border-radius:7px}
    table{width:100%;min-width:680px;border-collapse:collapse;font-size:13px;line-height:1.5;overflow-wrap:normal}th,td{padding:11px 13px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
    th{background:#eaf2f7;color:#153c55}tr:nth-child(even){background:#f8fafc}tr:last-child td{border-bottom:0}
    ul,ol{padding-left:25px}li{margin:5px 0}hr{border:0;border-top:1px solid var(--line)}
    .print-button{background:#175e8a;border:0;color:white;padding:8px 13px;border-radius:5px;cursor:pointer;font:inherit;margin-bottom:18px}
    .kicker{text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:700;color:#52758b}
    @media(max-width:960px){.layout{display:block}aside{position:static;height:auto;padding:20px}aside .toc{max-height:260px;overflow:auto}main{padding:25px 22px}h1{font-size:30px}}
    @media print{body{background:white;font-size:10pt}.layout{display:block}aside{display:none}main{padding:0;border:0}h1{font-size:24pt}h2{font-size:17pt;break-after:avoid}h3{break-after:avoid}.table-wrap{overflow:visible}pre{white-space:pre-wrap;background:#f0f3f5;color:#111}img{break-inside:avoid}a{color:inherit}}
    """
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>IE643 - Complete Project and Intensive Assessment Guide</title>'
                f'<style>{css}</style></head><body><div class="layout"><aside>'
                '<p class="kicker">IE643 / Study guide</p><button class="print-button" onclick="window.print()">Print / Save as PDF</button>'
                + nav + '</aside><main>' + str(soup) + '</main></div></body></html>')
    path = REPORT / "IE643_ASSESSMENT_GUIDE.html"
    path.write_text(document, encoding="utf-8")
    print(md_path)
    print(path)
    print(f"Words: {len(source.split()):,}; embedded figures: {len(soup.find_all('img'))}; tables: {len(soup.find_all('table'))}")


if __name__ == "__main__":
    main()
