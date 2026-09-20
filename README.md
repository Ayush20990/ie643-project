# Unsupervised Domain Adaptation for Multivariate Time-Series Forecasting
*IE 643 Course Project*

A lightweight multivariate forecaster (**LiteTCN**) is trained on **Domain 1** and adapted, **without any Domain 2 labels**, to **Domain 2** with **CARE-DA** (Correlation-Aligned, Reconstruction-regularised, Elastic-anchored Domain Adaptation). Three benchmarks, each with two distinct domains:

| Benchmark | Domain 1 (source) | Domain 2 (target) | Variables | L → H |
|---|---|---|---|---|
| Climate (Open-Meteo ERA5, hourly 2021-24) | Shimla, 2195 m (hilly) | New Delhi, 214 m (plain) | temperature, humidity, dew point, surface pressure, wind, cloud cover, radiation | 168 h → 24 h |
| Finance (Yahoo Finance, NSE daily 2012-25) | IT basket (TCS, INFY, WIPRO, HCLTECH, TECHM) | Pharma basket (SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, LUPIN) | log-return, high-low range, overnight gap, log-volume | 60 d → 5 d |
| Energy (ETT, hourly 2016-18) | ETTh1 (transformer, county 1) | ETTh2 (transformer, county 2) | HUFL, HULL, MUFL, MULL, LUFL, LULL, OT | 168 h → 24 h |

## Repository layout
```
src/
  config.py          dataset definitions, paths, window sizes
  data_download.py   step 1: download + curate all datasets  -> data/raw, data/processed
  data.py            chronological split, per-domain scaling, windowing
  model.py           LiteTCN-Forecaster (RevIN + dilated Conv1d encoder + linear path + recon head)
  losses.py          MK-MMD, Deep CORAL, DANN (gradient reversal), L2-SP anchor
  metrics.py         MSE, MAE, MASE, key-variable RMSE, gap-closure ratio
  train.py           source training, CARE-DA + baselines + ablations
  run_experiments.py full benchmark for one dataset (3 seeds) -> results/<ds>, models/<ds>
  plots.py           all figures -> results/figures
app/app.py           Streamlit interface (Domain 1 and Domain 2 upload, plots, metrics)
models/<ds>/         source_model.pt (Domain 1), adapted_model.pt (Domain 2, CARE-DA), meta.json (scalers, config)
results/<ds>/        train_log.txt, histories.json (per-epoch adaptation logs), metrics_all.csv, summary_target.csv
sample_inputs/       ready-to-upload CSVs from the held-out test period of each domain
presentation/        prep-presentation builder (build_deck.js) and the .pptx
report/              stage-wise report (.docx) and its builder
```

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate      # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```
Python 3.10+ is required. Everything runs on a CPU; each benchmark takes roughly 20–60 min per seed on a laptop.

## Reproduce
```bash
python -m src.data_download                             # 1. data (≈2 min)
python -m src.run_experiments --dataset climate --seeds 0 1 2   # 2. train + adapt + evaluate
python -m src.run_experiments --dataset energy  --seeds 0 1 2
python -m src.run_experiments --dataset finance --seeds 0 1 2
python -m src.sensitivity --dataset climate              # (optional) post-hoc λ sensitivity, also energy / finance
python -m src.adapt_v2                                  # 2b. CARE-DA v2 from saved Domain-1 checkpoints
python -m src.plots                                     # 3. figures
streamlit run app/app.py                                # 4. interface
```

Rebuilding the slides and the report (needs Node.js with `pptxgenjs react react-dom react-icons sharp docx` installed):
```bash
python presentation/collect_numbers.py   # results -> presentation/numbers.json
node presentation/build_deck.js          # -> presentation/IE643_Prep_Presentation.pptx
node report/build_report.js              # -> report/IE643_Project_Report.docx (open in Word, update the table of contents)
```
Team name and roll numbers are set at the top of `build_deck.js` and `build_report.js`.

## Using the interface
1. `streamlit run app/app.py`, open the printed URL.
2. Sidebar: choose a benchmark and **Domain 1** (uses the source model) or **Domain 2** (uses the CARE-DA adapted model).
3. Upload a CSV with a `date` column and the variables listed under *Expected CSV format* (finance also accepts raw `Date, Open, High, Low, Close, Volume`). Without an upload, a bundled sample from `sample_inputs/` is used.
4. Move the slider to pick the look-back window. The plot shows the **input**, the **actual** future and the **forecast** for each selected variable (Domain 2 also overlays the un-adapted source model).
5. Metrics: window-level MSE / MAE / MASE / key-variable RMSE, and a whole-file rolling evaluation against source-only and seasonal-naive baselines. Forecasts can be downloaded as CSV.
6. The *Benchmark results* tab shows the full results table and figures.

## Method in one paragraph
The source model is trained with forecasting MSE plus a masked-reconstruction auxiliary loss. **CARE-DA v1** (the designed objective) was

`L_v1 = MSE_src + λ_M·MMD²(z_s, z_t) + λ_C·CORAL(z_s, z_t) + λ_R·MaskedRecon(x_t) + λ_A·‖θ − θ_src‖²`

Ablations and a λ_R sweep showed that masked reconstruction *hurts* target forecasts on every benchmark. **CARE-DA v2** (the shipped adapted model) therefore drops that term and the near-zero CORAL term, and adds a forecast-consistent inner-horizon pretext, input-view consistency, multi-scale MMD and spectral alignment:

`L_v2 = MSE_src + λ_M·MMD²(z_s, z_t) + λ_H·MMD²(h_s, h_t) + λ_I·InnerForecast(x_t) + λ_K·Consist(x_t) + λ_S·Spectral(x_s, x_t) + λ_A·‖θ − θ_src‖²`

Only target *input* windows are used. v2 also picks the checkpoint that minimises `source_val_MSE + feature_MMD` (no target labels).

## Metrics
* **MSE / MAE** on per-domain z-scored values; **MASE** against a seasonal-naive forecast (scale-free); **RMSE of the key variable** in physical units.
* **Gap-closure ratio** `GCR = (E_src-only − E_method)/(E_src-only − E_oracle)`, where the oracle is the source model fine-tuned *with* target labels. This is the share of the supervised gap closed without labels.
* **Feature MMD²** between domains (was alignment achieved?), **forgetting %** (change in Domain 1 error after adaptation), and mean ± std over 3 seeds.

## Data sources
* Open-Meteo Historical Weather API (ERA5 reanalysis): https://open-meteo.com/en/docs/historical-weather-api
* Yahoo Finance via `yfinance`: https://github.com/ranaroussi/yfinance
* ETT dataset: https://github.com/zhouhaoyi/ETDataset
