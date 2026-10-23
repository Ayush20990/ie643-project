# Unsupervised Domain Adaptation for Multivariate Time-Series Forecasting
*IE 643 Course Project*

## Get the complete project

Team **DeepGen**: Ayush Patel (23B0037) and Rishik Yadav (23B2103).

- [Complete download and individual report / PPT / dashboard files](https://github.com/Ayush20990/ie643-project/releases/latest)
- [Assessment presentation](presentation/intensive/IE643_Intensive_Assessment.pptx)
- [Technical report](report/intensive/IE643_Intensive_Assessment_Report.pdf)
- [Complete study guide](report/IE643_ASSESSMENT_GUIDE.md)
- [Code recording script](recording/CODE_WALKTHROUGH.html) and [dashboard demo script](recording/DASHBOARD_DEMO.html)

The repository includes application code, all saved model families, sample inputs,
raw/processed datasets, experiment results and plots, the midterm and intensive
presentations, and report sources. Download the release ZIP for the packaged handoff,
or clone the repository to work together. The app runs locally; GitHub hosts the files.

```powershell
git clone https://github.com/Ayush20990/ie643-project.git
cd ie643-project
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-experiments.txt
.venv/Scripts/python -m streamlit run app/app.py
```

Tested with Python 3.13.2 and CPU PyTorch 2.6.0 on Windows. Use the local URL printed
by Streamlit. Saved checkpoints are included, so training is unnecessary for the demo.
Choose **Next hour** or **Next 24 hours** at the top for climate/energy; finance uses
five trading days. Use a bundled sample or upload a CSV. Opening the offline HTML
dashboard shows saved results; run the Streamlit app for predictions from a CSV.

For a third-city upload check, use the [Pune weather CSV](sample_inputs/weather_pune_2025-10-01_2025-10-31.csv)
or its [history-only version](sample_inputs/weather_pune_2025-10-01_2025-10-31_history_only.csv).
The [external weather guide](sample_inputs/EXTERNAL_WEATHER_README.md) explains
both modes, units and provenance. These examples use the existing checkpoints;
uploading does not retrain a city-specific model.

Run verification with `.venv/Scripts/python -m unittest discover -s tests -v`.
To rebuild the packaged handoff locally, run `.venv/Scripts/python -m src.package_submission`.

**Intensive assessment (Team DeepGen, updated 23 October 2026):**
[technical report and LaTeX](report/intensive/), [intensive slides and notes](presentation/intensive/),
[offline dashboard](dashboard/IE643_Results_Dashboard.html).
The reviewer-facing CSV app is `python -m streamlit run app/app.py`.
Choose **Prediction type** at the top of the app: **Next hour** or **Next 24 hours**
for weather/energy. The selection changes the trained model, forecast timestamps,
plot, historical evaluation and benchmark evidence. Upload one CSV and switch
between the tasks; each uses its own checkpoint. Finance retains five trading days.
The current submission suite passes **47 tests**, including **36 submission** and
**76 archived research UI combinations**; see `results/submission/validation.log`.
The earlier full experiment explorer is preserved as `app/research.py`.
The packaged handoff is available from the release linked above. Running the package
builder creates `submission/DeepGen_IE643_CourseProject_IntensiveAssessment/` locally.
See [handoff details](submission/README.md) for the download and remaining team tasks.

**Recording scripts:** open the HTML files in [recording/](recording/) in a browser.
For the code video, Rishik covers data/models (0:00–4:35), then Ayush covers
adaptation/inference (4:35–9:15). For the demo, Ayush covers weather, CSV upload
and rolling evaluation (0:00–2:30), then Rishik covers day-ahead, energy and finance
(2:30–4:45). Each guide includes spoken paragraphs, screen actions, a rehearsal
clock and a printable view. The code guide includes verified source line references.

**Assessment preparation:** [complete project study guide (HTML)](report/IE643_ASSESSMENT_GUIDE.html)
or [editable Markdown](report/IE643_ASSESSMENT_GUIDE.md). Covers the original work,
all later experiment stages, architecture/losses, verified results and plots,
demo/code walkthrough, an 18-slide plan and 32 viva questions.

## Latest: rolling next-hour forecasting and peak-focused adaptation

The project has no required horizon; the user approved adding **rolling next-hour
forecasts** alongside the retained **single-origin 24-hour benchmark**. Every rolling
point uses newly observed history. It is not a day-ahead forecast and must not be
compared with 24-hour RMSE as if only the model had changed.

The final peak-calibrated hybrid combines a new small **LiteTCN**, local dynamics,
the earlier hybrid and persistence using causal trend regimes. It predicts all
seven variables. Calibration emphasizes maxima/minima and ramps in daily rolling
episodes wholly inside each observed target **X**. No original target forecast
**Y**, actual future drivers or cross-window label reconstruction enters adaptation
or selection. Source replay/anchors, ordinary-error guards and fallback remain active.

| Target temperature, mean of 3 training seeds, same 1-hour task | Source-only RMSE | Adapted RMSE | Persistence RMSE | Reduction vs persistence |
|---|---:|---:|---:|---:|
| Climate, degC | 0.6791 | 0.6530 | 1.2692 | 48.55% |
| Energy oil temperature, degC | 1.2653 | 0.9151 | 1.3334 | 31.37% |

The final calibration reduces peak/trough MAE versus the first new next-hour hybrid:
climate **0.447/0.654 -> 0.281/0.551 degC**, energy **0.859/0.734 ->
0.848/0.665 degC**. Energy persistence still has lower trough error (**0.476 degC**),
and the first next-hour hybrid has slightly lower energy temperature RMSE
(**0.8970 degC**). The earlier climate hybrid's first-hour output remains competitive
(RMSE **0.6540 degC**, peak MAE **0.2622 degC**). No complete peak/dip solution is claimed.
All models remain selectable; no test-based per-variable winner policy was fitted.

The separate geometry-aware CatBoost/LiteTCN 24-hour adaptation gives climate
**1.3136 degC** and energy **4.0168 degC** temperature RMSE: modest improvements over
the retained TCN (**1.11% / 2.01%**), without beating the earlier strongest overall
families. Finance retains its original strict input-only, five-trading-day model.
These reused development periods are not independent confirmation.

[Current technical notes, all comparisons and plots](report/IE643_Forecast_Horizons_Results.html)
include fixed first/middle/last rolling segments, matched-horizon references,
event metrics, source/adaptation behavior and model size/CPU latency. **40 tests
passed**, including **76 UI combinations**. Artifact audits verify checkpoint,
data and training-code hashes, selection guards and full deployment source/target
MSE. A new TCN has 8,903 parameters; the full peak-calibrated hybrid has 110,840
neural parameters and about 0.7 MB of serialized tensors (including expert buffers).
This technical evidence does not replace the team's official-template assessment report.

```bash
pip install -r requirements-experiments.txt
python -m src.nowcast_train
python -m src.nowcast_hybrid_train
python -m src.peak_nowcast_train
python -m src.event_boost_train
python -m src.event_boost_ablation
python -m src.check_horizon_models
python -m unittest discover -s tests -v
python -m src.forecast_horizons_report
streamlit run app/app.py
```

Choose **Peak-calibrated next-hour adaptation** in the interface. With a CSV
containing subsequent actual observations, the app shows rolling predictions and
metrics. With only the lookback history, it outputs one next-hour forecast.
Use **24-hour peak-aware boosted adaptation** for the separate day-ahead experiment.
Checkpoints and logs are under `models/` and `results/` in `peak_next_hour`,
`next_hour_hybrid`, `next_hour` and `event_boost`. Exact code/data/initial-expert
hashes and selection-before-test manifests accompany the saved runs.

## Previous stage: predicted-driver decoder follow-up

Both requested directions are complete: gradual LiteTCN encoder unfreezing and
source-only temperature-driver analysis, followed by controlled small-decoder
experiments. The final decoder composes with the retained **input-adapted**
backbone and includes its exact zero-correction forecast as a candidate. All
target fitting/selection remains within individual observed inputs.

| Target development data, three seeds | Retained TCN MSE → composed MSE | Temperature RMSE → composed RMSE |
|---|---:|---:|
| Climate | 0.220609 → 0.220615 | 1.32844 → 1.32935 degC |
| Energy | 0.146226 → 0.146098 | 4.09915 → 4.08351 degC |

The energy gain is **small**: 0.087% MSE and 0.382% temperature RMSE. Climate is
effectively unchanged/slightly worse. These do not establish a peak fix or a new
best model; the existing hybrid/DLinear alternatives remain stronger overall.
No significance over independent time periods is claimed. Existing defaults and
checkpoints are preserved; choose **LiteTCN predicted-driver decoder** in the app
to inspect the composition. Finance is unchanged.

The correction head adds 2,689 parameters. It receives LiteTCN-predicted future
variables, observed history statistics and known calendar features. Actual future
weather/load values never become inputs. A temperature-only decoder control masks
additional non-temperature conditioning; its underlying TCN remains multivariate.
Source-validation benefits did not reliably transfer, so negative source-backbone
controls are retained under `models/results/coupled_litetcn`. Final composed
models/results are under `models/coupled_transfer` and `results/coupled_transfer`.

[Latest experiment notes and plots](report/IE643_Coupled_LiteTCN_Results.html)
include all controls, source/target metrics, peak/timing errors, 1/3/6/12/24-hour
diagnostics, training/selection logs and source-only driver comparisons. All
**29 tests passed**, including **56 UI combinations**, with checkpoint/code/data
hashes and full deployment MSE independently recomputed by artifact audits.
The October weather period was inspected before the decoder follow-up and is
not claimed as independent confirmation for these decoder experiments.

```bash
python -m src.coupled_train
python -m src.coupled_transfer
python -m src.climate_driver_analysis
python -m src.check_coupled
python -m src.check_coupled_transfer
python -m unittest discover -s tests -v
python -m src.coupled_report
```

## Gradual LiteTCN-R unfreezing follow-up

The requested follow-up is complete: matched head-only vs gradual encoder
unfreezing, three seeds for climate/energy, input-only adaptation, source-only
energy-load diagnostics, and a frozen-model check on new 1–17 October 2026 weather.
All checkpoints and negative controls are retained under `models/progressive_litetcn`.

| Target development benchmark | Previous retained TCN MSE / key RMSE | New TCN MSE / key RMSE |
|---|---:|---:|
| Climate | 0.22061 / 1.328 degC (v3) | 0.22838 / 1.386 degC |
| Energy | 0.14623 / 4.099 degC (R) | 0.14682 / 4.077 degC |

These results **do not establish a new best model**. Source validation rejects
late unfreezing checkpoints in favor of the initial/early head fit. Adaptation
exercises encoder updates but does not reliably improve the development forecast.
Energy key RMSE improves slightly over the retained TCN while overall MSE worsens;
the existing strict ensemble still has lower energy temperature RMSE (3.976 degC).
Existing model defaults are preserved; the new experiment is separately selectable.

Adding past load features to the source-only temperature/calendar ridge diagnostic
changes source-validation RMSE from 1.6875 to 1.6842 degC, only 0.194%. This is
limited predictive association, not a causal conclusion or proof of an error floor.
The independent October climate check has only 385 overlapping origins per domain:
new target MSE 0.16474 vs hybrid 0.16668, but temperature RMSE 1.983 vs hybrid 1.913
degC. It does not support a claim of better peaks.

Supporting notes and plots: [Gradual-unfreezing results](report/IE643_Progressive_LiteTCN_Results.html).
Metrics, histories, hashes, horizon/peak diagnostics and audits are under
`results/progressive_litetcn`. **27 tests pass**, including **52 UI combinations**.
The linear skip/RevIN stay fixed, and no target forecasting labels enter fitting
or selection. Training uses 24 sampled source rounds and 8 adaptation rounds,
at most 4,096 windows per round, not full dataset passes.

```bash
python -m src.progressive_train
python -m src.energy_driver_analysis
python -m src.progressive_confirm
python -m src.check_progressive
python -m unittest discover -s tests -v
python -m src.progressive_report
```

## Retained LiteTCN model experiments

LiteTCN remains part of the project. New source/adapted checkpoints are saved for
two revisions, with explicit comparisons against the stronger existing models:

| Dataset | Retained TCN variant | Original saved TCN MSE → new MSE | Key RMSE → new RMSE | Previous best MSE |
|---|---|---|---|---:|
| Climate | LiteTCN v3 | 0.23570 → 0.22061 | 1.474 → 1.328 degC | **0.21605** (hybrid) |
| Energy | LiteTCN-R | 0.15590 → 0.14623 | 4.183 → 4.099 degC | **0.14478** (DLinear family) |

These improve the original saved LiteTCN, but **do not beat the strongest existing
alternatives**. The table is a development comparison, not independent confirmation.
Keeping the alternatives is justified by the measured errors; the TCN family and
its failed/negative ablations remain documented for continuity with the midterm.
Existing finance models/results are retained; new hourly peak experiments cover
climate and energy.

**LiteTCN v3** uses 44,399 parameters, seasonal folding that preserves hourly
positions, depthwise dilated residual blocks, a channel-specific linear skip,
and optional calendar inputs. All 3 source controls share the same budget:
no-clock/MSE, clock/MSE, clock/ramp/extrema loss. Source validation chooses
clock/MSE on both datasets; the shape loss does not win the combined criterion.
Training uses 36 sampled rounds of at most 4,096 windows each, batch 256, AdamW
0.001 and cosine decay. These are not 36 full dataset passes.

**LiteTCN-R** retains the original source-trained three-layer encoder/projection
weights exactly. A DLinear-initialized shared linear skip gives a strong
zero-residual baseline, while a regularized TCN residual head is source fitted
and adapted. It has 229,902 parameters; only the 21,672 head parameters are updated.
On energy its neural residual slightly worsens total MSE versus the linear skip
alone. This negative result is included explicitly.

Adaptation and selection remain **strictly target-input-only**: withhold values
inside each input, use source replay/anchoring, enforce a source validation guard,
and include a fallback. No target forecasting labels or cross-window label
reconstruction are used. A separate optional recent-residual correction uses
observations inside each input and nested input-only validation.

The old model can fit peaks on a small source subset (energy RMSE 2.88 → 0.72 degC
on 64 examples). Full source/validation diagnostics show smoothing and overfitting;
missing future weather/load drivers and the MSE averaging objective are plausible
contributors to forecast errors. Simply making all peaks higher would worsen the
climate examples that already overshoot.

Frozen LiteTCN v3 models were also evaluated on **new Jan–Sep 2026 climate data**,
downloaded after freezing the weights/policies. On Delhi, temperature RMSE is
1.852 degC (1.834 with recent residuals), versus 1.949 for the original TCN and
1.768 for the previous hybrid. LiteTCN-R is not claimed as independently confirmed
on that period. Data/provenance are under `data/confirmatory_2026/climate` and
`results/peak_litetcn/climate_2026`.

```bash
python -m src.peak_train --datasets climate energy --seeds 0 1 2 --epochs 36 --adapt-epochs 10 --max-windows 4096 --batch-size 256 --threads 4
python -m src.peak_calibrate --datasets climate energy --seeds 0 1 2
python -m src.residual_train --datasets climate energy --seeds 0 1 2
python -m src.peak_confirm
python -m src.check_peak
python -m src.peak_report
python -m unittest discover -s tests -v
streamlit run app/app.py
```

These commands reuse the earlier source checkpoints reproduced below. New models
are under `models/peak_litetcn`, `models/peak_calibrated`, and
`models/residual_litetcn`. The app offers all three with explicit names. Calendar
models require consecutive hourly timestamps. Default TCN choices demonstrate
project continuity; benchmark tables retain stronger alternatives.

All **24 tests passed**, including **48 interface combinations**. Artifact audits
verify model/data hashes, selection rules, retained encoder weights and deployment
metrics. CPU measurements show that the smaller v3 model uses less storage but is
slower per window; parameter count alone is not an inference-speed metric.

[Model results and fixed comparison plots](report/IE643_LiteTCN_Model_Results.html)
are supporting experimental material. The official assessment template was verified
against the supplied ZIP and saved unchanged in `report/official_template`.
The team's final report should use that template and its own verified narrative.

## Earlier strict label-free refinement

The follow-up adds **variable-specific ridge forecasting heads** and a guarded
convex ensemble of those heads, Patch MLP and DLinear. Each target input window is
processed independently: adaptation withholds values **inside that window**.
Target forecasting labels and reconstruction of labels from overlapping windows
are excluded from both training and selection. A source-validation guard and a
source-model fallback limit negative transfer.

| Dataset | Previous best MSE | Strict ensemble MSE | Shorter-context MSE |
|---|---:|---:|---:|
| Climate | 0.22247 | 0.21743 | **0.21605** |
| Energy | **0.14478** (DLinear) | 0.14560 | 0.14556 |
| Finance | 0.54751 | **0.53728** | 0.53733 |

Climate's shorter-context model improves MSE by **2.9%** and temperature RMSE from
**1.334 to 1.284 degC**. Finance improves MSE by **1.9%** and now beats input mean
(0.54604) by **1.6%**. Energy's strict ensemble improves oil-temperature RMSE from
4.075 to **3.976 degC**, but its total MSE is slightly worse than DLinear; retain
DLinear when total multivariate energy MSE is the priority. These tables average
three existing neural training seeds; the new ridge fits are deterministic.

A separate **2025 climate evaluation**, downloaded after freezing the first strict
ensemble, confirms a target MSE reduction from **0.18897 to 0.18390 (2.7%)** versus
the previous Patch MLP. No 2025 data was used for fitting or selection. Original
and newly downloaded December 2024 context overlaps match exactly for all variables.
The shorter-context follow-up is exploratory and is not claimed as independently
confirmed on this new period. Energy and finance still use the development test.

```bash
python -m src.strict_refinement --datasets climate energy finance --seeds 0 1 2 --threads 4
python -m src.strict_context --datasets climate energy finance --seeds 0 1 2 --threads 4
python -m src.confirm_climate
python -m src.check_strict
python -m src.strict_report
python -m unittest discover -s tests -v
streamlit run app/app.py
```

These commands require the earlier checkpoints from the experiments below.
Install `requirements-experiments.txt` for the recorded package versions, including
SciPy for constrained mixture calibration. New models/results are saved separately
under `models/strict_refined`, `models/strict_context`, and matching results folders.
Select **Strict label-free ensemble** (default) or **Strict shorter-context ensemble**
in the interface; prior models remain selectable. Both source and target CSV input,
plots, actual/forecast metrics and downloads are supported.

See [the latest quantitative report](report/IE643_Strict_Refinement_Report.html),
`results/strict_refined/headline.csv`, the per-seed selection records and artifact
audit. CPU costs are measured in `results/strict_refined/efficiency.csv`; combining
multiple small models costs more than inference with one component.

The broader-history pilots in `models/refined`, `results/refined` and the incomplete
`history_refined` folders are **excluded** from strict results. They predate the
user's strict label-free constraint and are marked `EXCLUDED_FROM_STRICT_RESULTS`.
Their reconstruction of forecasting pairs across overlapping windows is not used
by either strict runner, the interface or the new climate-period evaluation.

## Current experiments (19 October 2026)

The new pipeline compares **3-, 5-, and 7-layer LiteTCNs**, a **patch MLP**, and a
**RevIN + DLinear-inspired baseline**, with the same forecasting-only source loss
and training budget. Architecture selection uses source validation MSE. Adaptation
uses source replay, forecasting withheld portions of observed target inputs, an
L2-SP anchor, and optionally MMD. Checkpoint selection uses source validation and
target **input** validation; it includes the source checkpoint as a fallback.
Target test futures are evaluated after selection.

```bash
python -m src.improve --datasets climate energy finance --seeds 0 1 2 --epochs 15 --adapt-epochs 8
python -m src.transfer_selection --datasets climate energy finance --seeds 0 1 2
python -m src.check_improved
python -m src.benchmark_inference
python -m src.improvement_report
python -m unittest discover -s tests -v
streamlit run app/app.py
```

Measured target MSE, averaged over three new training seeds, against the original
saved CARE-DA v2 seed-0 checkpoint:

| Dataset | Compact model | Original MSE | New MSE | Change |
|---|---|---:|---:|---:|
| Climate | Patch MLP | 0.2357 | 0.2225 | -5.6% |
| Energy | DLinear family | 0.1559 | 0.1448 | -7.1% |
| Finance | Patch MLP, initial selector | 0.5520 | 0.5475 | -0.8% |

Climate temperature RMSE improves from 1.474 to 1.334 degC. Most gains come from
architecture choice; DLinear adaptation adds little on energy. Finance still loses
to the input-mean baseline (MSE 0.5460). These are development-benchmark comparisons;
the follow-up needs a fresh time period or domain for external validation.
In the interface, choose **Validation-selected experiments** for the initial Patch
MLP runs or **DLinear transfer experiment** to inspect the linear alternative.

The training command resumes completed candidate checkpoints and seeds with matching
settings. It writes `models/improved/<dataset>/` and `results/improved/<dataset>/`,
preserving all original experiments. The interface offers **Validation-selected
experiments** when a bundle is available and **Original CARE-DA v2** for comparison.
The first requested seed is the deployment checkpoint, chosen independently of test
scores. See [the stage update](report/IE643_Improvement_Report.html), generated after
training, and `results/improved/headline.csv` for the measured outcomes and limitations.

The follow-up `src.transfer_selection` adapts DLinear too and compares it with the
initial selected model using a common source-validation + target-input forecasting
proxy. It saves separate `models/transfer_selection/<dataset>/` bundles, exposed as
**Selected using target history** in the interface. Its comparison tables are under
`results/transfer_selection/`. This extension was motivated by the initial development
test findings; it is explicitly exploratory, with no claim of an independent new
holdout. It still excludes target future values from training and checkpoint/model
selection. A new time period or domain is needed for external confirmation.
The interface also exposes **DLinear transfer experiment**, whose adapter is chosen
by the same input-only proxy within that model family. This makes a strong compact
alternative available even when the cross-model proxy ranks another model first.

An audit found that the original v2 spectral term compared raw inputs and therefore
had no model gradient. The implementation now aligns learned encoder features;
the old checkpoints still reflect the original implementation. The new adaptation
recipes do not use this term. New tables include `MASE_train` with fixed per-variable
training denominators; historical `MASE` used a per-window denominator. Gap closure
is undefined if the supervised reference does not improve on source-only.

The original report and presentation below describe historical runs. The new HTML
report records the controlled depth comparison, architecture choices, adaptation
losses, test results, plots, data inventory, interface preparation and remaining
submission steps. TA approval of the three dataset choices remains a team action.

## Original experiments

The legacy reconstruction and adversarial recipes assume a LiteTCN encoder. Use the
new runners above for the alternative architectures.

A lightweight multivariate forecaster (**LiteTCN**) is trained on **Domain 1** and adapted, **without any Domain 2 labels**, to **Domain 2** with **CARE-DA** (Correlation-Aligned, Reconstruction-regularised, Elastic-anchored Domain Adaptation). Three benchmarks, each with two distinct domains:

| Benchmark | Domain 1 (source) | Domain 2 (target) | Variables | L → H |
|---|---|---|---|---|
| Climate (Open-Meteo historical API, hourly 2021-24) | Shimla, 2195 m (hilly) | New Delhi, 214 m (plain) | temperature, humidity, dew point, surface pressure, wind, cloud cover, radiation | 168 h → 24 h |
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
