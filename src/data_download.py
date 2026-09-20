"""Download and pre-process the three domain-adaptation benchmarks.

Usage:  python -m src.data_download

Outputs  data/raw/...            untouched downloads
         data/processed/<ds>/<domain>.csv   columns: date, [series], <variables>
"""
import io
import time

import numpy as np
import pandas as pd
import requests

from .config import (CLIMATE_RANGE, CLIMATE_SITES, DATASETS, ETT_URL, FINANCE_RANGE,
                     FINANCE_TICKERS, PROC_DIR, RAW_DIR)


def _save(df, ds, domain):
    out = PROC_DIR / ds
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{domain}.csv", index=False)
    print(f"  saved {ds}/{domain}.csv  shape={df.shape}")


# ------------------------------------------------------------------ climate (Open-Meteo ERA5 archive)
def download_climate():
    cols = DATASETS["climate"]["columns"]
    for site, loc in CLIMATE_SITES.items():
        params = {**loc, "start_date": CLIMATE_RANGE[0], "end_date": CLIMATE_RANGE[1],
                  "hourly": ",".join(cols), "timezone": "Asia/Kolkata"}
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=120)
        r.raise_for_status()
        js = r.json()
        (RAW_DIR / "climate").mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "climate" / f"{site}.json").write_text(r.text)
        df = pd.DataFrame(js["hourly"]).rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df[cols] = df[cols].interpolate(limit_direction="both")
        print(f"climate/{site}: elevation={js.get('elevation')} m")
        _save(df[["date"] + cols], "climate", site)


# ------------------------------------------------------------------ finance (Yahoo Finance, NSE)
def ohlcv_to_features(df):
    """OHLCV -> stationary features used by the model (all in %, except log volume)."""
    f = pd.DataFrame({"date": pd.to_datetime(df["Date"] if "Date" in df else df["date"])})
    c, o, h, l, v = (df[k].astype(float).values for k in ["Close", "Open", "High", "Low", "Volume"])
    prev_c = np.r_[np.nan, c[:-1]]
    f["log_return"] = 100 * np.log(c / prev_c)
    f["hl_range"] = 100 * np.log(h / l)
    f["gap"] = 100 * np.log(o / prev_c)
    f["log_volume"] = np.log1p(v)
    f["close"] = c
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f[(f["log_volume"] > 0) & (f["hl_range"] >= 0)]


def download_finance():
    import yfinance as yf
    for domain, tickers in FINANCE_TICKERS.items():
        frames = []
        for t in tickers:
            for attempt in range(4):
                raw = yf.download(t, start=FINANCE_RANGE[0], end=FINANCE_RANGE[1],
                                  progress=False, auto_adjust=True)
                if len(raw):
                    break
                time.sleep(5 * (attempt + 1))
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index()
            (RAW_DIR / "finance").mkdir(parents=True, exist_ok=True)
            raw.to_csv(RAW_DIR / "finance" / f"{t}.csv", index=False)
            feat = ohlcv_to_features(raw)
            feat.insert(1, "series", t)
            frames.append(feat)
            print(f"finance/{domain}/{t}: {len(feat)} days")
        _save(pd.concat(frames, ignore_index=True), "finance", domain)


# ------------------------------------------------------------------ energy (ETT benchmark)
def download_energy():
    for name, domain in [("ETTh1", "etth1"), ("ETTh2", "etth2")]:
        r = requests.get(ETT_URL.format(name=name), timeout=120)
        r.raise_for_status()
        (RAW_DIR / "energy").mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "energy" / f"{name}.csv").write_text(r.text)
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["date"])
        _save(df, "energy", domain)


if __name__ == "__main__":
    download_climate()
    download_energy()
    download_finance()
