"""Build two offline, timed recording scripts from the current project code.

Documentation only. Does not fit models, change checkpoints or operate the app.
"""
import hashlib
import html
import json
import re
from pathlib import Path

from .config import ROOT


def focus(file, anchor, count=12, before=0):
    path = ROOT / file
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    matches = [i for i, line in enumerate(lines) if anchor in line]
    if not matches:
        raise ValueError(f'Missing recording anchor: {file}: {anchor}')
    start = max(0, matches[0]-before)
    end = min(len(lines), start+count)
    return dict(file=file, anchor=anchor, start=start+1, end=end,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                excerpt='\n'.join(f'{i+1:>3}  {lines[i]}' for i in range(start, end)))


def step(start, end, speaker, title, actions, speech, refs=(), essential=''):
    seconds = lambda v: int(v.split(':')[0])*60+int(v.split(':')[1])
    return dict(start=start, end=end, start_seconds=seconds(start), end_seconds=seconds(end),
                speaker=speaker, title=title, actions=actions, speech=speech,
                refs=list(refs), essential=essential)


CODE = [
    step('0:00','0:25','Rishik','Introduce the task and the two speakers',
         ['Show the repository in the IDE. Keep both face tiles visible.',
          'Open src/config.py. Do not explain the folder tree line by line.'],
         'Hello, we are team DeepGen: Rishik Yadav, 23B2103, and Ayush Patel, 23B0037. Our IE643 project adapts a lightweight multivariate forecaster from one domain to another. I will explain the data and forecasting models. Ayush will explain adaptation, the final hybrid, and inference.'),
    step('0:25','1:10','Rishik','The three tasks and their data',
         ['Point at DATASETS in src/config.py: the source, target, columns, L and H fields.',
          'Briefly open ohlcv_to_features in src/data_download.py. Point at its four feature formulas.'],
         'The three domain pairs are Shimla to Delhi weather, transformer station ETTh1 to ETTh2, and IT to pharma company time series. Weather and energy predict all seven variables. Energy temperature means transformer oil temperature. Finance predicts returns, high-low range, opening gap and log volume, rather than stock price directly. Weather comes from Open-Meteo, energy from ETT, and finance from cached Yahoo Finance downloads. The original horizons are 24 hours and five trading days. The new one-hour task applies to weather and energy.',
         [focus('src/config.py','DATASETS =',34),focus('src/data_download.py','def ohlcv_to_features',15)],
         'Three domain pairs; all variables; finance has engineered features and a different horizon.'),
    step('1:10','2:05','Rishik','Chronological splits, scaling and windows',
         ['In src/data.py point at chrono_split, then make_windows.',
          'In prepare point at Scaler.fit(parts["train"]). Show series_list only briefly for finance.'],
         'We split each series chronologically, so training precedes validation and testing. Weather and finance use a seventy-ten-twenty split, and energy uses sixty-twenty-twenty. Finance company series remain separate when creating windows. Each domain has a scaler fitted on its training period. In make_windows, the first L observations become X and the following H observations become Y. Earlier observations can provide validation or test context, but they are not new future labels. For the strict target adaptation, the fitting routines receive observed X; the original target Y is reserved for evaluation. The scaler and input-tail adaptation are separate operations.',
         [focus('src/data.py','def chrono_split',12),focus('src/data.py','def make_windows',11),focus('src/data.py','def prepare',18)],
         'Split by time and company; fit the scaler on training data; keep the original target Y out of adaptation.'),
    step('2:05','2:45','Rishik','The original LiteTCN and the reason for revisions',
         ['Open src/model.py at LiteTCNForecaster.',
          'Point at self.revin, self.enc, self.head and self.linear; then the addition in forward.'],
         'This is the LiteTCN we started with. RevIN normalizes each observed window. Temporal convolutions extract patterns, and the forecast combines a neural head with a linear temporal path. A reconstruction head supported the original adaptation experiments. We tested deeper networks and alternatives, but extra layers did not consistently improve the results. We therefore kept LiteTCN in the project and moved toward forecasting-aligned self-supervision and complementary predictors. The final hourly result belongs to a hybrid, rather than to this original network alone.',
         [focus('src/model.py','class LiteTCNForecaster',21),focus('src/model.py','def forward(self, x, return_z=False, return_h=False):',12)],
         'LiteTCN was retained; deeper layers alone did not resolve the errors.'),
    step('2:45','3:50','Rishik','The small next-hour LiteTCN',
         ['Open src/nowcast_tcn.py. Show the constructor, then the last return in forward.',
          'Point at context=48, H=1, dilations 1/2/4 and zero initialization.'],
         'The new next-hour network uses the latest 48 observed hours, differences, and six deterministic time features. It has width 32 and residual convolution blocks with dilations one, two and four. These blocks capture recent dynamics without the original strided compression. For seven variables this TCN has 8,903 trainable parameters. It predicts an increment relative to the latest observation, scaled by recent variability. The final layer starts at zero, so the initial prediction is exactly persistence. The clock features are known from timestamps, not observed future weather or load. This model outputs one future timestamp for all seven variables. A rolling curve is formed by repeating that prediction after each new actual observation arrives.',
         [focus('src/nowcast_tcn.py','class NextHourTCN',12),focus('src/nowcast_tcn.py','def forward(self,packed):',18)],
         'H=1 predicts increments from recent history; a rolling curve requires newly observed values.'),
    step('3:50','4:35','Rishik','Source training and optimizer settings',
         ['In src/nowcast_train.py show source_fit and the optimizer/scheduler lines.',
          'Point at train_loss in the loop and validation-based checkpoint selection. Do not run training.'],
         'Source training uses labeled source histories and futures. AdamW starts at learning rate 0.001, with weight decay 0.0001 and cosine decay. We compare ordinary and turning-weighted source controls, and use source validation to choose the source model. The recorded budget is 28 sampled rounds, with up to 4,096 windows per round; these are not full-dataset epochs. Adaptation later uses learning rate 0.0002 for twelve sampled rounds. We save the chosen checkpoints. Ayush will now show how adaptation uses target history while excluding its original forecast future.',
         [focus('src/nowcast_train.py','def source_fit',26)],
         'AdamW and cosine decay; source validation selects checkpoints; training is not part of the recording.'),
    step('4:35','5:40','Ayush','Strict within-input adaptation',
         ['Introduce yourself during the handoff.',
          'Show within_input in src/nowcast_tcn.py, then adapt_fit in src/nowcast_train.py.',
          'Point at the actual loss line and the source-validation eligibility guard.'],
         'I am Ayush Patel. The key boundary is in within_input. The final observed hour becomes a practice answer, and the predictor receives only the preceding history. For example, when 168 hours are already observed, an inner task uses hours one to 167 to predict observed hour 168. The real forecast then uses the full history to predict unseen hour 169. No original target forecast Y is passed to this adaptation function. Its loss combines target-history prediction, half-weighted source replay, and a 0.001 anchor toward source parameters. Source validation guards and the unchanged source checkpoint provide protection against harmful adaptation. The target historical values supply self-supervision; label-free does not mean that numerical target history is unavailable.',
         [focus('src/nowcast_tcn.py','def within_input',4),focus('src/nowcast_train.py','def adapt_fit',29)],
         'Practice labels are inside observed X; target original future Y stays excluded.'),
    step('5:40','6:30','Ayush','Five experts and causal trend regimes',
         ['Open src/peak_nowcast.py at experts, then forward.',
          'Point at [:,:1] for the earlier 24-hour head; briefly point at observed_regime.'],
         'The final model combines five outer forecasts: source-only and adapted versions of the new TCN-and-dynamics hybrid, source-only and adapted earlier-hybrid outputs, and persistence. From the earlier 24-hour head we take only the first hour, so every expert predicts the same timestamp. Nonnegative weights sum to one for each variable and trend regime. The regime uses recent observed slopes and acceleration; it does not inspect the future peak. This retains LiteTCN while allowing local dynamics and earlier predictors to contribute. The experts are frozen when the final mixture weights are calibrated.',
         [focus('src/peak_nowcast.py','def experts',14),focus('src/peak_nowcast.py','def observed_regime',13)],
         'Five same-horizon experts; regimes use past values; weights form a convex mixture.'),
    step('6:30','7:45','Ayush','Event loss, observed episodes and source safeguards',
         ['Show collect_episodes in src/peak_nowcast_train.py. Its arguments contain input_windows, not target Y.',
          'Point at EVENT_WEIGHT / RAMP_WEIGHT, then objective and constraints inside fit_weights.',
          'Keep the matrix algebra collapsed; explain its purpose only.'],
         'This collector creates a daily episode of successive one-hour practice forecasts entirely inside an already observed input. At each practice origin only the earlier prefix is visible. The event objective combines point MSE, twice the peak-and-trough error, and 0.1 times ramp error. Historical extrema identify the calibration loss locations, but never enter the regime features. This separates event supervision during fitting from causal inputs during prediction. The final calibration fits mixture weights with constrained SLSQP, rather than retraining all neural experts. It balances source and target-history objectives with a small source-weight anchor. Guards allow at most ten percent source point-error deterioration and five percent target-input point-error deterioration on those calibration data. Sparse regimes and failed fits fall back to source weights. These guards are not guarantees about unknown future test errors.',
         [focus('src/peak_nowcast_train.py','EVENT_WEIGHT=2.',3),focus('src/peak_nowcast_train.py','def collect_episodes',14),focus('src/peak_nowcast_train.py','objective=lambda',17)],
         'Observed episodes, event-sensitive loss and constrained frozen-expert calibration; no future-label regime features.'),
    step('7:45','8:35','Ayush','CSV inference and the horizon selector',
         ['In src/submission.py show load_selected and predict_window.',
          'Briefly show the Prediction type radio in app/submission_app.py. Do not explain layout code.'],
         'The app loads saved checkpoints according to dataset, prediction type and model. Weather and energy offer one hour or 24 hours; finance uses its existing strict hybrid at five trading days. CSV validation checks required columns and consecutive hourly timestamps. In predict_window, only rows before the chosen origin enter X. Predictions are computed before actual future rows are used for scoring. Changing the prediction option therefore changes the checkpoint and output horizon together. Uploaded files are used for frozen inference and evaluation, not for automatic retraining.',
         [focus('src/submission.py','def load_selected',17),focus('src/submission.py','def predict_window',15),focus('app/submission_app.py','horizon = st.radio',7)],
         'Mode selects a matching checkpoint; only preceding rows enter inference; actuals are for scoring.'),
    step('8:35','9:15','Ayush','Metrics, saved evidence and close',
         ['Show all_metrics in src/metrics.py, then peak_metrics in src/diagnose_peaks.py.',
          'Open results/submission/validation.log at its final test summary; do not rerun the suite.'],
         'Global MSE measures standardized error across all predicted variables. Key-variable RMSE reverses the scaling, giving temperature error in degrees Celsius or finance high-low-range error in log-percent units. We separately measure errors at actual peaks and troughs. Source-only and adapted models must be compared at the same horizon. The complete suite recorded 47 passing tests. The app deploys seed zero, while the main benchmark tables average three seeds. The periods were reused during development, so we do not claim independent confirmation. The separate demo shows the saved models working through the interface.',
         [focus('src/metrics.py','def all_metrics',10),focus('src/diagnose_peaks.py','def peak_metrics',18)],
         'Physical-unit RMSE, matched horizons, event metrics, three-seed benchmarks and honest limitations.'),
]


DEMO = [
    step('0:00','0:15','Ayush','Introduce the working interface',
         ['Show the running Forecast studio. Begin with climate, Next hour, Peak-calibrated hybrid and Domain 1 source.',
          'Use the bundled source sample; clear any earlier upload before recording.'],
         'This is DeepGen\'s forecasting interface. I am Ayush, and Rishik will cover the second half. We will show source and target inputs, both hourly and day-ahead modes, and finance.'),
    step('0:15','0:50','Ayush','Source example and CSV schema',
         ['Point at Dataset, Input domain and the bundled source filename.',
          'Open CSV requirements briefly. Show the input and next-hour point. Keep the input-only checkbox OFF.'],
         'Domain one uses the saved source model and source scaler. This example contains hourly multivariate weather observations. The CSV requirements list all seven variables. The chart displays temperature, but the model predicts all seven. A single next-hour forecast is one future timestamp. With actual future rows already in the file, the app can display an error for that selected window.'),
    step('0:50','1:20','Ayush','Upload the target-domain CSV',
         ['Change Input domain to Domain 2 target.',
          'Upload sample_inputs/climate_domain2_delhi.csv. Confirm the displayed filename and Delhi label.',
          'Keep Next hour and Peak-calibrated hybrid. Point at Actual future, Source-only and Adapted.'],
         'I now upload the Delhi target-domain CSV and select domain two. We compare source-only and adapted forecasts on the same target data. The gray curve is observed history; the other traces show actuals and predictions. The displayed metrics belong to this selected window, so they can differ from our full benchmark averages.'),
    step('1:20','2:05','Ayush','Rolling evaluation and downloadable evidence',
         ['Scroll to Evaluate historical forecasts.',
          'Set Maximum forecast origins to evaluate (latest available) to 96, then click Run historical evaluation.',
          'Show the result table and rolling curve. Click Download evaluated forecasts and actuals once.'],
         'Historical evaluation gives a clearer comparison across many origins. Here we use the latest 96 available origins in this uploaded file. Every point predicts only the next hour, using history updated with the newly observed actual value. Model weights stay fixed. The table compares matching source-only, adapted and naive forecasts. We can export the actual values and predictions rather than relying only on the visual curve.'),
    step('2:05','2:30','Ayush','Forecast without actual future labels and hand over',
         ['Scroll back to Forecast studio and tick Forecast after the final uploaded row.',
          'Point at the forecast and the message that no actual future score is available.',
          'Click Download this forecast, then UNTICK the checkbox before the handoff.'],
         'The app can also forecast beyond the final uploaded row. Because the future actual is unavailable, it shows a prediction without inventing an error score. I will restore the historical view. Rishik will now show the longer horizon and the other datasets.'),
    step('2:30','3:10','Rishik','The separate 24-hour forecasting task',
         ['Say your name, then select Next 24 hours at the top.',
          'Keep climate, target domain and the same Delhi upload. Select Strict forecasting hybrid.',
          'Confirm the input-only checkbox is OFF. Show the 24-hour curve and expand View all 24 forecast timestamps and variables.'],
         'I am Rishik. This option loads the 24-hour forecasting model and predicts an entire day from one origin. It receives no new observations inside that horizon. The table contains 24 future timestamps for all seven variables. Timing and amplitude errors remain visible. We therefore report this harder task separately and do not describe lower one-hour error as a like-for-like day-ahead improvement.'),
    step('3:10','3:45','Rishik','Energy and the remaining trough limitation',
         ['Choose Energy, Domain 2 target, Next hour and Peak-calibrated hybrid.',
          'Use the automatically selected bundled ETTh2 sample. Keep OT as the displayed variable.',
          'Run historical evaluation with 96 origins if rehearsal loading time permits; otherwise show the forecast and the saved event evidence.'],
         'The same workflow works for the transformer dataset. OT means transformer oil temperature, and all seven load-and-temperature variables are forecast. The hourly benchmark improves overall oil-temperature error, but persistence remains better at the highlighted trough metric. We retain that limitation rather than claiming the peak and dip problem is completely solved.'),
    step('3:45','4:25','Rishik','Finance, including its weak results',
         ['Choose Finance, Domain 2 target and Strict forecasting hybrid. Its only prediction type is Next 5 trading days.',
          'Use the bundled SUNPHARMA OHLCV sample and point at hl_range.',
          'Open Benchmark dashboard and show the target strict_adapted, strict_source and input_mean rows, comparing the MSE column.'],
         'Finance predicts four engineered features, including the high-low range shown here, for five trading days. It does not predict stock prices directly. Forecasts can stay close to a mean and miss movements. In the saved target benchmark, adapted global MSE is about 0.5373 versus 0.5460 for input mean. That is only a 1.60 percent gain, and adaptation adds about 0.14 percent over the source-only hybrid. Uploaded-window scores are separate.'),
    step('4:25','4:45','Rishik','Close with limitations and reproducibility',
         ['Open Method & downloads. Show the limitation notice and report/study-guide downloads.',
          'Finish the narration by 4:45; reserve the remaining 15 seconds for natural clicks/transitions.'],
         'The method page explains observed-input adaptation and the remaining limitations. Reports and supporting evidence are downloadable. These are saved, frozen models, with separately labeled tasks and baselines. Both source and target domains can be tested through the same CSV interface. Thank you.'),
]


def cue(before, text, file=None, anchor=None):
    """Attach an unspoken direction to an exact point in the existing narration."""
    ref = focus(file, anchor, 1) if file else None
    if ref:
        text = text.replace('{location}', f'{file}:{ref["start"]}')
    return dict(before=before, text=text, ref=ref)


# Cues are separate from speech so they never inflate the spoken word count.
# Resolve every source location on rebuild rather than maintaining line numbers.
CODE_CUES = [
    [cue('Hello,', 'Keep the IDE and both face tiles visible. Rishik begins.')],
    [cue('The three domain pairs', 'Switch to {location} → DATASETS. Point at source/target, columns, L and H.',
         'src/config.py', 'DATASETS ='),
     cue('Finance predicts', 'Switch to {location} → ohlcv_to_features(). Point at the four feature formulas.',
         'src/data_download.py', 'def ohlcv_to_features'),
     cue('The original horizons', 'Return to {location} → DATASETS. Point at the L and H fields.',
         'src/config.py', 'DATASETS =')],
    [cue('We split', 'Switch to {location} → chrono_split(). Point at the chronological array slices.',
         'src/data.py', 'def chrono_split'),
     cue('Finance company', 'Stay in data.py; jump to {location} → series_list(). Point at grouping by series.',
         'src/data.py', 'def series_list'),
     cue('Each domain', 'Jump to {location} → prepare(). Point at Scaler.fit(parts["train"]).',
         'src/data.py', 'scaler = Scaler.fit(parts["train"])'),
     cue('In make_windows,', 'Jump to {location} → make_windows(). Point at xs.append(w[:, :L]) and ys.append(w[:, L:]).',
         'src/data.py', 'def make_windows')],
    [cue('This is the LiteTCN', 'Switch to {location} → LiteTCNForecaster.__init__(). Point at self.revin and self.enc.',
         'src/model.py', 'class LiteTCNForecaster'),
     cue('and the forecast combines', 'Jump to {location} → forward(). Point at the neural-head + linear-path addition.',
         'src/model.py', 'yn = self.head(z)'),
     cue('A reconstruction head', 'Return to {location} → self.recon in the constructor. Keep this explanation brief.',
         'src/model.py', 'self.recon =')],
    [cue('The new next-hour', 'Switch to {location} → NextHourTCN.__init__(). Point at context=48, width=32, H=1 and dilations (1,2,4).',
         'src/nowcast_tcn.py', 'class NextHourTCN'),
     cue('It predicts an increment', 'Jump to {location} → forward(). Point at last + sd * self.head(inputs).',
         'src/nowcast_tcn.py', 'return last+sd*self.head'),
     cue('The final layer starts', 'Scroll back to {location} → the two nn.init.zeros_ calls.',
         'src/nowcast_tcn.py', 'nn.init.zeros_'),
     cue('The clock features', 'Jump to {location} → forward(). Point at the timestamp-based future-feature loop.',
         'src/nowcast_tcn.py', 'for j,period in enumerate(TIME_PERIODS)')],
    [cue('Source training', 'Switch to {location} → source_fit(). Point at AdamW and the cosine scheduler.',
         'src/nowcast_train.py', 'def source_fit'),
     cue('We compare ordinary', 'Scroll to {location} → source_fit() validation/eligibility checks and best checkpoint update.',
         'src/nowcast_train.py', 'eligible=q["MSE"]'),
     cue('The recorded budget', 'Stay in source_fit(); point at the sampled-window selection and training loop.',
         'src/nowcast_train.py', 'model.train();ix=torch.randperm'),
     cue('Ayush will now', 'Handoff: Rishik finishes. Ayush takes over at 4:35; keep both face tiles visible.')],
    [cue('I am Ayush', 'Ayush begins. Switch to {location} → within_input(). Keep this short function visible for the example.',
         'src/nowcast_tcn.py', 'def within_input'),
     cue('The final observed hour', 'Point at {location}: packed[:,:-1] is the predictor; packed[:,-1:] is the observed practice answer.',
         'src/nowcast_tcn.py', 'return packed[:,:-1]'),
     cue('No original target forecast', 'Switch to {location} → adapt_fit(). Point at target_train_x / target_val_x in its signature, then within_input().',
         'src/nowcast_train.py', 'def adapt_fit'),
     cue('Its loss combines', 'Scroll to {location} → the loss assignment. Point at target loss, +.5 source replay and +.001 anchor.',
         'src/nowcast_train.py', 'loss=forecast_loss(model(xt[i])'),
     cue('Source validation guards', 'Scroll to {location} → eligible and the best checkpoint update.',
         'src/nowcast_train.py', 'eligible=sv["MSE"]')],
    [cue('The final model', 'Switch to {location} → PeakNowcast.experts(). Point at the five entries in p.',
         'src/peak_nowcast.py', 'def experts'),
     cue('From the earlier', 'Point at {location} → [:,:1], which selects the first hour of the earlier head.',
         'src/peak_nowcast.py', 'reference=self.reference.components'),
     cue('Nonnegative weights', 'Scroll to {location} → forward(). Point at regime-dependent weights and the weighted sum.',
         'src/peak_nowcast.py', 'def forward(self,x)'),
     cue('The regime uses', 'Jump to {location} → observed_regime(). Point at current, previous and acceleration.',
         'src/peak_nowcast.py', 'def observed_regime')],
    [cue('This collector', 'Switch to {location} → collect_episodes(). Point at input_windows, episode_prefix() and the observed tail.',
         'src/peak_nowcast_train.py', 'def collect_episodes'),
     cue('The event objective', 'Scroll up to {location} → EVENT_WEIGHT=2 and RAMP_WEIGHT=.1.',
         'src/peak_nowcast_train.py', 'EVENT_WEIGHT=2.'),
     cue('Historical extrema', 'Jump to {location} → design(). Point at y.argmax / y.argmin: these choose loss locations only.',
         'src/peak_nowcast_train.py', 'w[i,y.argmax(1)]'),
     cue('The final calibration', 'Jump to {location} → fit_weights(). Point at minimize(..., method="SLSQP"); skip matrix algebra.',
         'src/peak_nowcast_train.py', 'fit=minimize(objective'),
     cue('It balances source', 'Point at {location} → objective: source/target event losses and source-weight anchor.',
         'src/peak_nowcast_train.py', 'objective=lambda'),
     cue('Guards allow', 'Point at {location} → the two inequality constraints, 1.1 source and 1.05 target-input point error.',
         'src/peak_nowcast_train.py', 'constraints=constraints+[{"type":"ineq","fun":lambda v:1.1'),
     cue('Sparse regimes', 'Point at {location} → min(ns[r],nt[r])<64, then the if not accepted fallback below.',
         'src/peak_nowcast_train.py', 'if min(ns[r],nt[r])<64')],
    [cue('The app loads', 'Switch to {location} → load_selected(). Point at the horizon/model dispatch.',
         'src/submission.py', 'def load_selected'),
     cue('CSV validation', 'Jump to {location} → read_input(). Point at the required columns and consecutive-hour validation.',
         'src/submission.py', 'def read_input'),
     cue('In predict_window,', 'Jump to {location} → predict_window(). Point at values[origin-l:origin] and the prediction call before truth.',
         'src/submission.py', 'def predict_window'),
     cue('Changing the prediction', 'Switch to {location} → main(). Point at the Prediction type radio and its horizon options.',
         'app/submission_app.py', 'horizon = st.radio')],
    [cue('Global MSE', 'Switch to {location} → all_metrics(). Point at MSE/MAE over all variables.',
         'src/metrics.py', 'def all_metrics'),
     cue('Key-variable RMSE', 'Stay in all_metrics(); point at the inverse scaling and key_RMSE calculation below.'),
     cue('We separately measure', 'Switch to {location} → peak_metrics(). Point at actual peak/trough indices and their errors.',
         'src/diagnose_peaks.py', 'def peak_metrics'),
     cue('The complete suite', 'Switch to {location} → the final test summary. Show 47 tests / OK; do not run the suite.',
         'results/submission/validation.log', 'Ran 47 tests'),
     cue('The app deploys', 'Keep the saved test summary visible and finish the interpretation. No more file switches.')],
]
for code_step, cues in zip(CODE, CODE_CUES, strict=True):
    code_step['cues'] = cues


def script_html(s):
    """Interleave visual directions without modifying the spoken narration."""
    parts, cursor = [], 0
    for c in s.get('cues', []):
        position = s['speech'].find(c['before'], cursor)
        if position < cursor:
            raise ValueError(f'Missing or out-of-order inline cue: {c["before"]}')
        parts.append('<span class="narration">'+html.escape(s['speech'][cursor:position])+'</span>')
        parts.append('<span class="cue" aria-label="Unspoken screen direction">['+html.escape(c['text'])+']</span>')
        cursor = position
    parts.append('<span class="narration">'+html.escape(s['speech'][cursor:])+'</span>')
    return ''.join(parts)


PREP = {
    'code': [
        'Target 9:15 of planned content, leaving 45 seconds inside the maximum 10 minutes. Rishik speaks 0:00-4:35; Ayush speaks 4:35-9:15.',
        'Pin the files listed in the timeline before recording. Enable editor line numbers and use readable text, around 17-20 px. Ctrl+P opens a file; Ctrl+G jumps to the listed line.',
        'Bracketed cues appear between the spoken lines. Follow them silently: switch files, jump to the indicated line, or point at the named function. They remain visible in Spoken lines only mode and in print; do not read them aloud.',
        'Record the IDE only, with both face tiles visible. Keep this guide on a second device, printout or non-shared screen. Read the spoken script; the click instructions are not narration.',
        'Rehearse once with the real mouse movements. The clock is a rehearsal aid, not a substitute for timing the exported recording.',
        'Do not run download, training or full tests during the video. Open the existing validation log and saved results. Focus on the displayed functions, not every loop or experiment file.',
        'This is a speaking allocation. Actual contributions remain: both researched and experimented; Ayush handled the app/dashboard; Rishik handled the report/PPT. Both should understand the method.',
    ],
    'demo': [
        'Target 4:45 planned content plus 15 seconds for real transitions, within a five-minute demo. Ayush speaks 0:00-2:30; Rishik speaks 2:30-4:45.',
        'Launch from the repository root: python -m streamlit run app/app.py. Use the URL printed by Streamlit. If the dashboard is already running at localhost:8502, use that tab.',
        'Before recording, rehearse every dataset/mode once so checkpoints and sample files are loaded. Start the actual video back at climate / Next hour / Peak-calibrated hybrid / Domain 1 source, with no uploaded file.',
        'Keep sample_inputs/climate_domain2_delhi.csv ready for the single live upload. If you choose to upload a different domain later, remove or replace the old CSV: changing the domain option does not change an already uploaded file.',
        'A recorded CSV example needs real future rows to show scores. The bundled samples contain those. Keep Forecast after the final uploaded row OFF except during the explicit input-only demonstration.',
        'Use 96 latest origins for the rolling display. These are scores on the demonstration CSV, not the whole benchmark, and are not promised to equal the report averages.',
        'If evaluation is slow, avoid rerunning it. Keep the actual upload/prediction, horizon switch and finance explanation. Use the clearly labeled Benchmark dashboard for saved results.',
        'Record the working Streamlit app for CSV inference. The offline HTML dashboard shows saved results and cannot run new CSV predictions.',
    ],
}


CAUTIONS = [
    'Say observed-input self-supervision: the inner practice answer is already inside past X. Do not claim that target numerical observations are never used.',
    'Name the final model a hybrid containing LiteTCN. Do not attribute the whole hybrid result to the original LiteTCN alone.',
    'A daily episode of 24 rolling H=1 predictions refreshes observations each hour. It is different from one H=24 forecast.',
    'Event loss fits frozen-expert mixture weights in the final calibration. The standalone TCN adaptation uses target pretext loss, source replay and an anchor.',
    'Weather plots emphasize air temperature; energy plots emphasize transformer oil temperature. Both predict seven variables. Finance predicts four features at H=5 trading days.',
    'Three-seed saved means, deployed seed 0 and uploaded-file scores are different quantities. MSE is not a percentage accuracy.',
    'Finance gain over input mean is about 1.60% in global MSE; adaptation over matching source-only adds about 0.14%. This does not establish price forecasting or trading profitability.',
    'Day-ahead peaks, energy troughs and reused development-test periods remain limitations. Do not promise every example or metric improves.',
]


STYLE = '''
*{box-sizing:border-box}body{margin:0;background:#f2f6f5;color:#1b2b31;font:17px/1.65 system-ui,Segoe UI,sans-serif}header{background:#16232a;color:white;padding:35px max(24px,calc((100vw - 1100px)/2))}header p{color:#d7e8e2;max-width:950px}h1{font:700 38px/1.2 Georgia,serif;margin:10px 0}h2{font:700 28px/1.3 Georgia,serif;margin:30px 0 15px}h3{font-size:22px;margin:0 0 8px}main{max-width:1150px;margin:auto;padding:25px}a{color:#246f66}nav{display:flex;gap:20px;flex-wrap:wrap}header a{color:#b6e2d7}.toolbar{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #cfdad8;padding:12px 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}button{border:1px solid #afc6bf;background:white;color:#1b2b31;padding:8px 14px;border-radius:4px;font:inherit;font-size:15px;cursor:pointer}#clock{font-weight:800;font-variant-numeric:tabular-nums;font-size:23px}#current{font-size:14px;color:#4f5f65}.intro,.segment,.notes{background:#fff;padding:24px 30px;margin:22px 0;border-left:4px solid #cfdad8}.segment.active{border-left-color:#d98a1e;box-shadow:0 0 0 2px #f1d4a8}.segment .meta{color:#2f7f73;font-weight:750;font-size:17px}.segment[data-speaker=Rishik] .meta{color:#55518c}.actions{color:#4f5f65}.actions p{margin:5px 0}.say{font:23px/1.7 Georgia,serif;margin:15px 0;padding:16px 0 6px;border-top:1px solid #e2eae7;white-space:pre-line}.wordcount{font-size:13px;color:#7c8b90}details{margin-top:18px}summary{cursor:pointer;font-weight:650}pre{font:13px/1.6 Consolas,monospace;overflow:auto;background:#f3f6f5;padding:15px;white-space:pre}code{font-family:Consolas,monospace;font-size:.9em}table{border-collapse:collapse;width:100%;font-size:15px}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce6e1;vertical-align:top}th{background:#e6f0ee}li{margin-bottom:8px}.note{color:#4f5f65}.essential{font-size:16px;color:#2f7f73}.speech-only .actions,.speech-only details,.speech-only .essential,.speech-only .prep,.speech-only .wordcount{display:none}.large .say{font-size:30px}footer{padding:25px 0;color:#647a71;font-size:14px}@media(max-width:700px){h1{font-size:29px}.say{font-size:21px}main{padding:12px}.segment,.intro,.notes{padding:20px 18px}.toolbar{position:static}}@media print{body{background:white;font-size:11pt}.toolbar,nav,details{display:none}header{background:white;color:#1b2b31;padding:0}header p{color:#4f5f65}main{padding:0}.intro,.segment,.notes{border:0;border-top:1px solid #ccc;box-shadow:none!important;padding:12px 0;break-inside:avoid}.say{font-size:13pt;line-height:1.5}h1{font-size:25pt}h2{font-size:19pt}td,th{padding:5px}a{color:#1b2b31}footer{font-size:9pt}}
'''


STYLE += '.cue{display:block;font:650 16px/1.55 system-ui,Segoe UI,sans-serif;color:#275f75;background:#eaf3f8;border-left:3px solid #4b8299;padding:9px 13px;margin:15px 0;white-space:normal}.large .cue{font-size:19px}@media print{.cue{font-size:10pt;background:#f1f5f7;break-after:avoid}}'


def document(kind, steps):
    esc = html.escape
    title = 'Code walkthrough script' if kind == 'code' else 'Dashboard demo script'
    other = 'DASHBOARD_DEMO.html' if kind == 'code' else 'CODE_WALKTHROUGH.html'
    rows = ''.join(f'<tr><td><a href="#s{i}">{s["start"]}-{s["end"]}</a></td><td>{s["speaker"]}</td><td>{esc(s["title"])}</td></tr>' for i,s in enumerate(steps,1))
    sections = []
    for i,s in enumerate(steps,1):
        refs = []
        for ref in s['refs']:
            refs.append(f'<p><code>{esc(ref["file"])}</code> - lines {ref["start"]}-{ref["end"]}. Search: <code>{esc(ref["anchor"])}</code></p><pre>{esc(ref["excerpt"])}</pre>')
        details = '<details><summary>Exact code to point at (verified against current files)</summary>'+''.join(refs)+'</details>' if refs else ''
        actions = ''.join(f'<li>{esc(a)}</li>' for a in s['actions'])
        words = len(s['speech'].split())
        sections.append(f'''<section class="segment" id="s{i}" data-start="{s['start_seconds']}" data-end="{s['end_seconds']}" data-speaker="{s['speaker']}"><div class="meta">{s['start']}-{s['end']} · {s['speaker']}</div><h3>{esc(s['title'])}</h3><div class="actions"><strong>Do on screen</strong><ol>{actions}</ol></div><div class="say">{script_html(s)}</div><div class="wordcount">Spoken text: {words} words. Read the narration; bracketed cues and action lists are silent directions.</div>{'<p class="essential"><b>Keep this point:</b> '+esc(s['essential'])+'</p>' if s['essential'] else ''}{details}</section>''')
    prep = ''.join('<li>'+esc(a)+'</li>' for a in PREP[kind])
    cautions = ''.join('<li>'+esc(a)+'</li>' for a in CAUTIONS)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepGen - {title}</title><style>{STYLE}</style></head><body><header><small>DEEPGEN · IE643 · RECORDING PREPARATION</small><h1>{title}</h1><p>Ayush Patel (23B0037) and Rishik Yadav (23B2103). A timed speaking plan with exact screen actions. Rehearse the explanation in your own voice.</p><nav><a href="{other}">Open the other video script</a><a href="#timeline">Speaker timeline</a><a href="#cautions">Important wording</a></nav></header><div class="toolbar"><span id="clock">00:00</span><button id="start">Start rehearsal</button><button id="reset">Reset</button><button onclick="document.body.classList.toggle('speech-only')">Spoken lines only</button><button onclick="document.body.classList.toggle('large')">Larger script</button><button onclick="window.print()">Print / save as PDF</button><span id="current">Ready</span></div><main><section class="intro prep"><h2>Before recording</h2><ol>{prep}</ol><p class="note">The timer highlights the current segment. Keep the guide off the recorded screen. These are scripts, not completed recordings or measured guarantees of speaking duration.</p></section><section class="intro" id="timeline"><h2>Who speaks when</h2><table><thead><tr><th>Time</th><th>Speaker</th><th>Show / explain</th></tr></thead><tbody>{rows}</tbody></table></section>{''.join(sections)}<section class="notes prep" id="cautions"><h2>Wording that keeps the explanation accurate</h2><ul>{cautions}</ul><p><b>If asked about earlier alternatives:</b> depth controls, Patch MLP / DLinear references and retained-TCN revisions are documented in the assessment guide. Explain the purpose of those experiments; do not open every training file in the ten-minute recording.</p><p><b>If asked about SLSQP:</b> it fits nonnegative mixture weights under sum-to-one and validation-error constraints. Neural AdamW adaptation and final mixture calibration are separate optimization stages.</p></section><footer>Generated from current source anchors and saved evidence on 24 October 2026. Rebuild with <code>python -m src.build_recording_guides</code>. Both recordings are per team. The code video maximum is 10 minutes; the demo is five minutes. Both members participate, and faces must be visible in the code walkthrough under the supplied assessment instructions.</footer></main><script>let elapsed=0,started=0,tick=null;const clock=document.getElementById('clock'),start=document.getElementById('start'),current=document.getElementById('current');const steps=[...document.querySelectorAll('.segment')];function draw(){{const s=elapsed+(tick?(Date.now()-started)/1000:0);clock.textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(Math.floor(s%60)).padStart(2,'0');let active=null;for(const e of steps){{const on=s>=Number(e.dataset.start)&&s<Number(e.dataset.end);e.classList.toggle('active',on);if(on)active=e;}}current.textContent=active?active.dataset.speaker+' · '+active.querySelector('h3').textContent:'Planned content finished / transition buffer';}}start.onclick=()=>{{if(tick){{elapsed+=(Date.now()-started)/1000;clearInterval(tick);tick=null;start.textContent='Resume rehearsal';}}else{{started=Date.now();tick=setInterval(draw,250);start.textContent='Pause';}}draw();}};document.getElementById('reset').onclick=()=>{{if(tick)clearInterval(tick);tick=null;elapsed=0;start.textContent='Start rehearsal';draw();current.textContent='Ready';}};</script></body></html>'''


def main():
    out = ROOT/'recording';out.mkdir(exist_ok=True)
    for kind,steps,name in [('code',CODE,'CODE_WALKTHROUGH.html'),('demo',DEMO,'DASHBOARD_DEMO.html')]:
        for i,s in enumerate(steps):
            assert s['start_seconds'] == (steps[i-1]['end_seconds'] if i else 0)
            assert s['end_seconds'] > s['start_seconds']
            assert len(s['speech'].split()) <= (s['end_seconds']-s['start_seconds'])*2.6
        (out/name).write_text(document(kind,steps),encoding='utf-8')
        print(name,'planned seconds',steps[-1]['end_seconds'],'spoken words',sum(len(s['speech'].split()) for s in steps))
    (out/'source_map.json').write_text(json.dumps({'code_steps':CODE,'demo_steps':DEMO},indent=2),encoding='utf-8')
    (out/'README.md').write_text('''# DeepGen recording scripts

Open `CODE_WALKTHROUGH.html` and `DASHBOARD_DEMO.html` in a browser. Both work offline and include a rehearsal clock, speaker timeline, screen actions, spoken text and printable views. The code script also includes verified line references and excerpts, with bracketed file/function cues directly between the spoken lines. Follow the cues silently; do not read them aloud.

- Code: Rishik 0:00-4:35; Ayush 4:35-9:15. Leave 45 seconds within the maximum ten minutes.
- Demo: Ayush 0:00-2:30; Rishik 2:30-4:45. Reserve 15 seconds for transitions in the five-minute video.

Use both faces in the code recording, readable screen text and clear audio. Keep these scripts on another device or non-shared screen. Rehearse the real navigation. Do not run training or download data during recording. `source_map.json` contains the script text and exact source anchors; regenerate after source changes with `python -m src.build_recording_guides`.

The app is launched from the repository root with `python -m streamlit run app/app.py`. In the extracted complete package, `python START_APP.py` also works. Use the URL printed by the server. Initial demo state: climate, Next hour, Peak-calibrated hybrid, Domain 1 source, bundled sample, input-only checkbox off. Upload the Delhi sample after switching to the target domain.
''',encoding='utf-8')


if __name__ == '__main__':
    main()
