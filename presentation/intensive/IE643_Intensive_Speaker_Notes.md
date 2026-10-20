# DeepGen intensive assessment speaker notes

18 content slides + 2 reference slides. Planned talk/demo: 10:55. Rehearse within the 12-minute allowance.

## 1. Domain adaptation for multivariate forecasting (0:15)

Introduce DeepGen and the IE643 task. Both members researched methods and ran experiments. Ayush developed the dashboard/app; Rishik prepared the report and presentation.

Evidence: report/IE643_ASSESSMENT_GUIDE.md

## 2. Outline (0:15)

Follow the problem and the midterm model, then explain the changes. Separate the original day-ahead evidence and added next-hour evidence. Finish with app, limitations and next assessment plan.

Evidence: report/IE643_ASSESSMENT_GUIDE.md

## 3. Forecasting under domain shift (0:30)

Source supplies histories and forecast labels. Adaptation uses target histories only. The unseen future of the original target forecasting task stays excluded. Historical values inside X supply self-supervised answers.

Evidence: src/config.py; src/data.py; src/strict_models.py

## 4. Project workflow (0:25)

Collect domain pairs and split chronologically. Train on source labels, adapt only from observed target inputs with source safeguards, freeze before evaluation and deploy through CSV inference.

Evidence: src/data.py; src/nowcast_train.py; src/peak_nowcast_train.py; src/submission.py

## 5. Before the review: LiteTCN and CARE-DA (0:35)

Original model: RevIN, strided convolutions, a linear path and reconstruction head. CARE combines supervised forecasting, alignment, self-supervision and an anchor. These saved v2 results are fixed seed 0, not three-seed means.

Evidence: src/model.py; src/train.py; results/*/care_v2_compare.csv

## 6. After the review: depth alone was insufficient (0:30)

Deeper TCNs were tested alongside linear and Patch MLP controls. TCN revisions and geometry-aware boosting gave mixed or modest day-ahead gains. The final next-hour hybrid keeps a causal LiteTCN and complementary experts. H=1 is an explicitly added task.

Evidence: results/improved; results/peak_litetcn; results/event_boost; results/peak_next_hour

## 7. Three datasets with distinct domains (0:40)

Weather comes from Open-Meteo. ETT contains transformer loads and oil temperature. Finance pools company series by sector and predicts engineered features. All configured variables are forecast.

Evidence: src/config.py; src/data_download.py; https://open-meteo.com/en/docs/historical-weather-api; https://github.com/zhouhaoyi/ETDataset

## 8. Observed history supplies the adaptation task (0:45)

Weather/finance splits 70/10/20, ETT 60/20/20. Scalers use each domain training period. All hours 1–168 are observed. The inner predictor hides and predicts the tail. Actual future Y stays excluded. For H=1, use 1–167 to predict observed 168; actual inference predicts 169. App inference uses fixed weights.

Evidence: src/data.py; src/strict_models.py; src/peak_nowcast_train.py

## 9. LiteTCN within the final next-hour hybrid (0:45)

The new causal TCN has 8,903 parameters. It predicts increments for seven variables from 48 observed hours plus clock features. Local dynamics is complementary. Five outer experts are source/adapted new hybrids, source/adapted earlier-hybrid first-hour outputs and persistence. Mixture weights depend on variable and causal regime.

Evidence: src/nowcast_tcn.py; src/nowcast_hybrid.py; src/peak_nowcast.py

## 10. Adaptation objective and safeguards (0:55)

TCN adaptation: target-input forecasting + 0.5 source replay + 0.001 relative anchor. Event calibration uses daily episodes within observed inputs. Extrema are loss labels only, never regime features. SLSQP fits convex weights. Source point guard 10%, target-input guard 5%, minimum 64 examples per regime and source fallback.

Evidence: src/nowcast_train.py; src/peak_nowcast_train.py

## 11. Experimental setup and metrics (0:30)

Three seeds measure training randomness. Deployment is seed 0. TCN budgets are sampled rounds, not full epochs. Compare matching source-only/adapted checkpoints. Development test periods were inspected during iterations, so they are not independent confirmation.

Evidence: results/next_hour/*/manifest.json; results/peak_next_hour/*/manifest.json; results/forecast_horizons/efficiency.csv

## 12. Original-horizon results (0:50)

Target three-seed means. Climate earlier hybrid: shorter-context strict ensemble. Energy earlier hybrid: strict ensemble as temperature-focused reference. Guarded DLinear has lower global MSE 0.144778 but oil RMSE 4.074870. Retained vs boosted is a whole-system comparison, not adaptation-only. Finance stays H=5. Right side compares global MSE with seasonal naive for hourly datasets and input mean for finance.

Evidence: presentation/intensive/evidence.json; results/strict_context; results/strict_refined; results/forecast_horizons/twenty_four_hour_headline.csv

## 13. 24-hour forecasts from a single origin (0:45)

First fixed target origin, seed 0, without visual performance filtering. Each curve is issued at once without updates inside the 24-hour horizon. Earlier temperature-focused hybrid remains a reference. Boosting lowers retained-TCN temperature RMSE only 1.11% climate and 2.01% energy and does not beat the earlier hybrid. Right-side values are three-seed whole-test RMSE, not the plotted-window error. Peak timing/amplitude errors remain.

Evidence: results/event_boost/*/forecast_examples.npz; results/strict_context/climate/forecast_examples.npz; results/strict_refined/energy/forecast_examples.npz; presentation/intensive/evidence.json

## 14. Rolling next-hour results (0:45)

Predict one hour and refresh observations before the next. Matching family source/adapted RMSE gains: 3.84% climate, 27.68% energy. Full-system gains over persistence are 48.55% and 31.37%, including source learning and hybrid components. Earlier first-hour climate RMSE 0.654038 is nearly as good as final 0.653009. First TCN/dynamics energy RMSE 0.896999 beats final 0.915108 but has worse event errors.

Evidence: results/forecast_horizons/next_hour_headline.csv; results/forecast_horizons/*_next_hour_comparison.csv

## 15. Hourly trajectories and peak tradeoffs (0:45)

First 96 fixed target origins, seed 0. Successive next-hour predictions, not a 96-hour forecast from one origin. Refresh inputs every hour with fixed weights. Side event numbers average three seeds over complete daily episodes. Final climate peak/trough MAE 0.2810/0.5514, energy 0.8477/0.6650. Energy persistence trough MAE 0.4761 remains better. First-hybrid vs final comparison changes experts/regimes and loss together.

Evidence: results/peak_next_hour/*/rolling_forecasts.npz; results/forecast_horizons/*_next_hour_peak_comparison.csv

## 16. CSV interface and live demonstration (0:55)

Open Streamlit, select a target CSV and show source/adapted forecasts, actuals and metrics. Switch H=1/H=24 and explain the different tasks. Demonstrate oil temperature and financial feature names. Selected-window scores differ from saved whole-test means. Model weights are fixed. The offline dashboard is a results viewer; the app performs CSV inference.

Evidence: app/submission_app.py; src/submission.py; dashboard/IE643_Results_Dashboard.html; local app screenshot, 20 October 2026

## 17. Conclusions and team contributions (0:25)

Three domain pairs, compact source/adapted models, input-only adaptation, comparative logs/plots and an interface are complete. LiteTCN stays in the hybrid. Hourly accuracy improves, while day-ahead peaks and energy troughs remain open. Roles follow the supplied team statement.

Evidence: report/intensive/IE643_Intensive_Assessment_Report.pdf; report/team_details.json

## 18. Plan for the novelty assessment (0:25)

Plans, not completed results. Lock a fresh chronological period. Hold experts fixed to isolate event loss, then isolate regime conditioning. Investigate energy troughs and uncertainty before distillation.

Evidence: report/intensive/report_body.tex; report/IE643_ASSESSMENT_GUIDE.md

## 19. References: forecasting and adaptation (Reference)

Papers motivate components and baselines. Custom implementations are not exact reproductions. Full bibliography is also in report/intensive/references.bib.

Evidence: report/intensive/references.bib

## 20. References: data, code and evidence (Reference)

Website publication dates are distinct from access dates. n.d. marks a missing stated publication year. The project is the local October 2026 edition; remote synchronization is not implied. Cached files and numerical sources have hashes in the report and evidence export.

Evidence: report/intensive/references.bib; src/config.py; presentation/intensive/evidence.json
