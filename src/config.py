"""Central configuration for the three source -> target domain-adaptation benchmarks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SAMPLE_DIR = ROOT / "sample_inputs"

# Each benchmark: a source domain (Domain 1) and a target domain (Domain 2) sharing the same
# variables. L = look-back window, H = forecast horizon, season = seasonal period for MASE.
DATASETS = {
    "climate": {
        "title": "Climate: Hilly (Shimla) -> Plain (Delhi)",
        "source": "shimla", "target": "delhi",
        "source_label": "Shimla (hilly, ~2200 m)", "target_label": "New Delhi (plain, ~215 m)",
        "columns": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
                    "wind_speed_10m", "cloud_cover", "shortwave_radiation"],
        "key": "temperature_2m", "key_unit": "degC",
        "L": 168, "H": 24, "season": 24, "split": (0.7, 0.1, 0.2), "train_stride": 2,
    },
    "finance": {
        "title": "Finance: IT stocks -> Pharma stocks (NSE)",
        "source": "it", "target": "pharma",
        "source_label": "NSE IT basket (TCS, INFY, WIPRO, HCLTECH, TECHM)",
        "target_label": "NSE Pharma basket (SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, LUPIN)",
        "columns": ["log_return", "hl_range", "gap", "log_volume"],
        "key": "hl_range", "key_unit": "% (daily high-low range)",
        "L": 60, "H": 5, "season": 1, "split": (0.7, 0.1, 0.2), "train_stride": 1,
    },
    "energy": {
        "title": "Energy: Transformer station ETTh1 -> ETTh2",
        "source": "etth1", "target": "etth2",
        "source_label": "ETTh1 (electricity transformer, county 1)",
        "target_label": "ETTh2 (electricity transformer, county 2)",
        "columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
        "key": "OT", "key_unit": "degC (oil temperature)",
        "L": 168, "H": 24, "season": 24, "split": (0.6, 0.2, 0.2), "train_stride": 1,
    },
}

CLIMATE_SITES = {
    "shimla": {"latitude": 31.1048, "longitude": 77.1734},
    "delhi": {"latitude": 28.6139, "longitude": 77.2090},
}
CLIMATE_RANGE = ("2021-01-01", "2024-12-31")

FINANCE_TICKERS = {
    "it": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
}
FINANCE_RANGE = ("2012-01-01", "2025-06-30")

ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
