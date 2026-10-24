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
         "Hi, we are team DeepGen: Rishik Yadav, 23B2103, and Ayush Patel, 23B0037. We train a model on one set of data, called the source. Then we adjust it for another set, called the target. I'll explain the data and training. Ayush will cover that adjustment and the app."),
    step('0:25','1:10','Rishik','The three tasks and their data',
         ['Point at DATASETS in src/config.py: the source, target, columns, L and H fields.',
          'Briefly open ohlcv_to_features in src/data_download.py. Point at its four feature formulas.'],
         "Let's start with the three datasets. For weather, we move from Shimla to Delhi. For energy, we move between two transformer stations. For finance, we move from IT companies to pharma companies. We predict all seven columns in weather and energy. The temperature in energy is transformer oil temperature. For finance, this function turns price and volume data into four columns: log return, high-low range, opening gap and log volume. Now, back in config.py, L tells us how much past data we use, and H tells us how far ahead we predict. Weather and energy have one-hour and 24-hour options. Finance uses five trading days.",
         [focus('src/config.py','DATASETS =',34),focus('src/data_download.py','def ohlcv_to_features',15)],
         'Three domain pairs; all variables; finance has engineered features and a different horizon.'),
    step('1:10','2:05','Rishik','Prepare the data for training',
         ['In src/data.py point at chrono_split, then make_windows.',
          'In prepare point at Scaler.fit(parts["train"]). Show series_list only briefly for finance.'],
         "Before training, we need to arrange this data properly. We first split it by time: older data for training, then validation, then testing. Weather and finance use a seventy-ten-twenty split; energy uses sixty-twenty-twenty. We also keep each company's data separate, so an input never mixes two companies. Next, we put the columns on a similar scale, using only the training period to work out the average and spread. Finally, we make pairs of past and future data. X is the past we give the model, and Y is what it should predict. We use source X and Y for training. For target adaptation, we only pass in X.",
         [focus('src/data.py','def chrono_split',12),focus('src/data.py','def make_windows',11),focus('src/data.py','def prepare',18)],
         'Split by time and company; fit the scaler on training data; keep the original target Y out of adaptation.'),
    step('2:05','2:45','Rishik','Our first model and why we changed it',
         ['Open src/model.py at LiteTCNForecaster.',
          'Point at self.revin, self.enc, self.head and self.linear; then the addition in forward.'],
         'Now that the inputs are ready, this is our original LiteTCN model. It first adjusts the level and scale of each input, then uses convolution layers to find patterns over time. Here, we add the neural prediction to a simpler linear prediction. This extra output was used in our earlier work to reconstruct parts of the input. We tried adding more layers, but that did not reliably fix the errors. So we kept LiteTCN and built a smaller version for the next-hour task. We also kept the separate 24-hour model.',
         [focus('src/model.py','class LiteTCNForecaster',21),focus('src/model.py','def forward(self, x, return_z=False, return_h=False):',12)],
         'LiteTCN was retained; deeper layers alone did not resolve the errors.'),
    step('2:45','3:50','Rishik','The small next-hour LiteTCN',
         ['Open src/nowcast_tcn.py. Show the constructor, then the last return in forward.',
          'Point at context=48, H=1, dilations 1/2/4 and zero initialization.'],
         "Here is the smaller next-hour model. It takes the latest 48 hours and predicts one hour ahead for all seven columns. There are three convolution blocks, and each looks at a different spacing in the history. The network has 8,903 trainable parameters. The prediction starts from the last observed value, then adds a change learned by the model. We start the final layer at zero. So, before training, it simply predicts that the next value will stay the same. We also give it time information, such as the time of day. For a rolling forecast, we predict one hour, receive the next actual reading, and predict again. Let's now see how we train it.",
         [focus('src/nowcast_tcn.py','class NextHourTCN',12),focus('src/nowcast_tcn.py','def forward(self,packed):',18)],
         'H=1 predicts increments from recent history; a rolling curve requires newly observed values.'),
    step('3:50','4:35','Rishik','Train the first model',
         ['In src/nowcast_train.py show source_fit and the optimizer/scheduler lines.',
          'Point at train_loss in the loop and validation-based checkpoint selection. Do not run training.'],
         "This function trains the first model using source data. We use AdamW to update its weights, starting with a learning rate of 0.001. That rate gradually gets smaller. We try two versions of the training error: one treats hours normally, and the other pays more attention when the direction changes. We use validation results to choose which saved model to keep. Training runs for 28 rounds, with up to 4,096 examples picked in each round. We now have a trained source model. Ayush will explain how we adjust it using the target's past data.",
         [focus('src/nowcast_train.py','def source_fit',26)],
         'AdamW and cosine decay; source validation selects checkpoints; training is not part of the recording.'),
    step('4:35','5:40','Ayush','Adjust the model using past target data',
         ['Introduce yourself during the handoff.',
          'Show within_input in src/nowcast_tcn.py, then adapt_fit in src/nowcast_train.py.',
          'Point at the actual loss line and the source-validation eligibility guard.'],
         "I'm Ayush. The source model is trained, but the target data may behave differently. We adjust it using a practice question from values we already have. This function holds out the last observed hour and predicts it from the earlier hours. If we have 168 hours, we use hours one to 167 to predict hour 168. The real forecast is still for hour 169. We then train a copy of the source model on these questions. The loss has three parts: target practice error, half the error on source examples so it keeps what it learned, and a small penalty for large weight changes. We also check performance on source validation data. If the changes aren't useful, we keep the original model. Next, we'll combine it with other forecasts.",
         [focus('src/nowcast_tcn.py','def within_input',4),focus('src/nowcast_train.py','def adapt_fit',29)],
         'Practice labels are inside observed X; target original future Y stays excluded.'),
    step('5:40','6:30','Ayush','Combine forecasts using recent trends',
         ['Open src/peak_nowcast.py at experts, then forward.',
          'Point at [:,:1] for the earlier 24-hour head; briefly point at observed_regime.'],
         'One model can still miss a rise or fall, so we combine forecasts. This function gives us five next-hour predictions. Two use our new TCN and recent-trend model, two use earlier models, and one repeats the last value. Each pair includes a source and an adapted version. From the earlier 24-hour models, we take only the first hour, so all predictions cover the same time. We then give each prediction a share in the final answer. These shares add up to one. Which shares we use depends on how the past values have been changing. Next, we need to learn the shares.',
         [focus('src/peak_nowcast.py','def experts',14),focus('src/peak_nowcast.py','def observed_regime',13)],
         'Five same-horizon experts; regimes use past values; weights form a convex mixture.'),
    step('6:30','7:45','Ayush','Learn the shares using past examples',
         ['Show collect_episodes in src/peak_nowcast_train.py. Its arguments contain input_windows, not target Y.',
          'Follow the data: collect_episodes builds practice examples, design scores them, and fit_weights learns the mixture.',
          'Keep the matrix algebra collapsed; explain its purpose only.'],
         "To learn these shares, we need examples where we know the answer. This function makes them from the last 24 hours of each past input. It predicts each hour from earlier hours and saves the five predictions, recent trend and actual answer. Here, we calculate their error. We add extra weight at the highest and lowest actual values, and a smaller term for getting the hour-to-hour change wrong. These past highs and lows help us score predictions; they don't tell the model about future peaks. We then choose shares that work on source data and target history, while staying close to the source shares. The solver shown here changes only the shares; the individual models stay the same. Finally, we check that errors stay within the allowed limits. If there are too few examples or the fit fails, we keep the source shares. Some future peaks can still be missed.",
         [focus('src/peak_nowcast_train.py','def collect_episodes',14),focus('src/peak_nowcast_train.py','def design',17),focus('src/peak_nowcast_train.py','objective=lambda',17)],
         'Observed episodes, event-sensitive loss and constrained frozen-expert calibration; no future-label regime features.'),
    step('7:45','8:35','Ayush','Use the saved models in the app',
         ['In src/submission.py show load_selected and predict_window.',
          'Briefly show the Prediction type radio in app/submission_app.py. Do not explain layout code.'],
         'Once the models are saved, we can use them in the app. This function loads the right model for the dataset and prediction length. First, we check the CSV for required columns and any missing hours in hourly data. Then we take only the rows before the time we want to predict. We make the prediction before reading future actuals for the error calculation. This option in the app selects one hour or 24 hours. Finance uses five trading days. Uploading a file uses the saved model; it does not train a new one.',
         [focus('src/submission.py','def load_selected',17),focus('src/submission.py','def predict_window',15),focus('app/submission_app.py','horizon = st.radio',7)],
         'Mode selects a matching checkpoint; only preceding rows enter inference; actuals are for scoring.'),
    step('8:35','9:15','Ayush','Check prediction errors and saved tests',
         ['Show all_metrics in src/metrics.py, then peak_metrics in src/diagnose_peaks.py.',
          'Open results/submission/validation.log at its final test summary; do not rerun the suite.'],
         "Finally, we check how good the predictions are. MSE and MAE measure error across all the columns after scaling. For temperature RMSE, we use degrees Celsius so the error is easier to understand. We also check errors at actual peaks and dips, since an average can hide mistakes. The saved test log shows 47 passing tests. The app uses one saved training run; the main results average three. We checked these test periods while improving the models, so they aren't a fresh, untouched test. That finishes our code walkthrough.",
         [focus('src/metrics.py','def all_metrics',10),focus('src/diagnose_peaks.py','def peak_metrics',18)],
         'Physical-unit RMSE, matched horizons, event metrics, three-seed benchmarks and honest limitations.'),
]


DEMO = [
    step('0:00','0:15','Ayush','Introduce the working interface',
         ['Show the running Forecast studio. Begin with climate, Next hour, Peak-calibrated hybrid and Domain 1 source.',
          'Use the bundled source sample; clear any earlier upload before recording.'],
         "Hi, I'm Ayush from team DeepGen. Rishik will join me for the second half. We'll show how to upload data, make hourly and 24-hour predictions, and check the results for all three datasets."),
    step('0:15','0:50','Ayush','Source example and CSV schema',
         ['Point at Dataset, Input domain and the bundled source filename.',
          'Open CSV requirements briefly. Show the input and next-hour point. Keep the input-only checkbox OFF.'],
         "We'll start with the weather data from our first region, Shimla. This uses the model trained on that region. Here are the seven columns the CSV needs. We're showing temperature on the chart, but the model predicts all seven columns. In next-hour mode, we get one future value for each column. Since this file also contains the actual next hour, we can compare the prediction with what happened."),
    step('0:50','1:20','Ayush','Upload the target-domain CSV',
         ['Change Input domain to Domain 2 target.',
          'Upload sample_inputs/climate_domain2_delhi.csv. Confirm the displayed filename and Delhi label.',
          'Keep Next hour and Peak-calibrated hybrid. Point at Actual future, Source-only and Adapted.'],
         "Now I'll switch to the second region and upload the Delhi CSV. We can compare the original model and the adapted model on the same data. The gray line shows the past input. The other lines show the actual future and our predictions. These scores are for this one prediction, so they may differ from the report's overall results."),
    step('1:20','2:05','Ayush','Rolling evaluation and downloadable evidence',
         ['Scroll to Evaluate historical forecasts.',
          'Set Maximum forecast origins to evaluate (latest available) to 96, then click Run historical evaluation.',
          'Show the result table and rolling curve. Click Download evaluated forecasts and actuals once.'],
         "One prediction doesn't tell us everything, so let's check several hours. I'll use the latest 96 available prediction times in this file. Each time, the model gets the updated actual history and predicts the next hour. It stays the same saved model throughout. This table compares the original and adapted models with simple baselines. We can also download the predictions and actual values to check them ourselves."),
    step('2:05','2:30','Ayush','Forecast without actual future labels and hand over',
         ['Scroll back to Forecast studio and tick Forecast after the final uploaded row.',
          'Point at the forecast and the message that no actual future score is available.',
          'Click Download this forecast, then UNTICK the checkbox before the handoff.'],
         "What if we only have past data? This option predicts after the file ends. We still get a prediction, but there is no actual future value here, so there is no error score. I'll turn that option off again. Rishik will now show the 24-hour mode and the other datasets."),
    step('2:30','3:10','Rishik','The separate 24-hour forecasting task',
         ['Say your name, then select Next 24 hours at the top.',
          'Keep climate, target domain and the same Delhi upload. Select Strict forecasting hybrid.',
          'Confirm the input-only checkbox is OFF. Show the 24-hour curve and expand View all 24 forecast timestamps and variables.'],
         "Hi, I'm Rishik. Here, we switch to predicting the next 24 hours. This loads a separate model and predicts the whole day from the same starting point. It doesn't receive new readings during those 24 hours. The table shows a prediction for every hour and all seven columns. You can still see mistakes around some peaks. This is a harder task than predicting one hour at a time, so we show its results separately."),
    step('3:10','3:45','Rishik','Energy and the remaining trough limitation',
         ['Choose Energy, Domain 2 target, Next hour and Peak-calibrated hybrid.',
          'Use the automatically selected bundled ETTh2 sample. Keep OT as the displayed variable.',
          'Run historical evaluation with 96 origins if rehearsal loading time permits; otherwise show the forecast and the saved event evidence.'],
         "Now let's try energy. This is data from a second transformer station. OT means oil temperature, which is the column shown here. Again, all seven columns are predicted. Our hourly model improves the overall temperature error, but simply repeating the last value still does better on the saved dip-error measure. So we haven't solved every peak and dip problem."),
    step('3:45','4:25','Rishik','Finance, including its weak results',
         ['Choose Finance, Domain 2 target and Strict forecasting hybrid. Its only prediction type is Next 5 trading days.',
          'Use the bundled SUNPHARMA OHLCV sample and point at hl_range.',
          'Open Benchmark dashboard and show the target strict_adapted, strict_source and input_mean rows, comparing the MSE column.'],
         "Finally, here's finance. We predict four values calculated from price and volume data, for five trading days. This chart shows the high-low range. We're not predicting the stock price itself. You can see that the model misses some movements. In the saved results, its overall MSE is about 0.5373, compared with 0.5460 for predicting the input average. That's only a 1.60 percent improvement. Adaptation adds about 0.14 percent over the original hybrid model. The scores for an uploaded file can be different."),
    step('4:25','4:45','Rishik','Close with limitations and reproducibility',
         ['Open Method & downloads. Show the limitation notice and report/study-guide downloads.',
          'Finish the narration by 4:45; reserve the remaining 15 seconds for natural clicks/transitions.'],
         'This page explains how the method works and where it still struggles. You can download the report and supporting material here. Both regions can be tested through the same upload option, using the saved models. That completes our demo. Thank you.'),
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
    [cue('Hi,', 'Keep the IDE and both face tiles visible. Rishik begins.')],
    [cue("Let's start", 'Switch to {location} ? DATASETS. Point at source/target and columns.',
         'src/config.py', 'DATASETS ='),
     cue('For finance,', 'Switch to {location} ? ohlcv_to_features(). Point at the four feature formulas.',
         'src/data_download.py', 'def ohlcv_to_features'),
     cue('Now, back in config.py,', 'Return to {location} ? DATASETS. Point at the L and H fields.',
         'src/config.py', 'DATASETS =')],
    [cue('We first split', 'Switch to {location} ? chrono_split(). Point at the time-ordered slices.',
         'src/data.py', 'def chrono_split'),
     cue("We also keep each company's", 'Stay in data.py; jump to {location} ? series_list(). Point at grouping by company.',
         'src/data.py', 'def series_list'),
     cue('Next, we put', 'Jump to {location} ? prepare(). Point at Scaler.fit(parts["train"]).',
         'src/data.py', 'scaler = Scaler.fit(parts["train"])'),
     cue('Finally, we make', 'Jump to {location} ? make_windows(). Point at xs.append(w[:, :L]) and ys.append(w[:, L:]).',
         'src/data.py', 'def make_windows')],
    [cue('Now that the inputs', 'Switch to {location} ? LiteTCNForecaster.__init__(). Point at self.revin and self.enc.',
         'src/model.py', 'class LiteTCNForecaster'),
     cue('Here, we add', 'Scroll to {location} ? forward(). Point at the neural-head + linear-path addition.',
         'src/model.py', 'yn = self.head(z)'),
     cue('This extra output', 'Point briefly at {location} ? self.recon in the constructor.',
         'src/model.py', 'self.recon =')],
    [cue('Here is the smaller', 'Switch to {location} ? NextHourTCN.__init__(). Point at context, H, width and the encoder.',
         'src/nowcast_tcn.py', 'class NextHourTCN'),
     cue('The prediction starts', 'Jump to {location} ? forward(). Point at last + sd * self.head(inputs).',
         'src/nowcast_tcn.py', 'return last+sd*self.head'),
     cue('We start the final layer', 'Scroll back to {location} ? the two nn.init.zeros_ calls.',
         'src/nowcast_tcn.py', 'nn.init.zeros_'),
     cue('We also give it time', 'Point at {location} ? the timestamp-based feature loop in forward().',
         'src/nowcast_tcn.py', 'for j,period in enumerate(TIME_PERIODS)')],
    [cue('This function trains', 'Switch to {location} ? source_fit(). Point at its inputs, NextHourTCN creation and optimizer.',
         'src/nowcast_train.py', 'def source_fit'),
     cue('We try two versions', 'Point at {location} ? forecast_loss() and the turning-control option.',
         'src/nowcast_train.py', 'loss=forecast_loss(model(x)'),
     cue('We use validation', 'Scroll to {location} ? the validation check and which saved model we keep.',
         'src/nowcast_train.py', 'eligible=q["MSE"]'),
     cue('Ayush will explain', 'Handoff: Rishik finishes. Ayush takes over at 4:35; keep both face tiles visible.')],
    [cue("I'm Ayush.", 'Ayush begins. First explain why adaptation is needed; then show the practice task.'),
     cue('This function holds out', 'Switch to {location} ? within_input(). Point at the earlier prefix and withheld last observation.',
         'src/nowcast_tcn.py', 'def within_input'),
     cue('We then train a copy', 'Switch to {location} ? adapt_fit(). Point at target X inputs, source copy and optimizer.',
         'src/nowcast_train.py', 'def adapt_fit'),
     cue('The loss has three parts', 'Scroll to {location} ? loss. Point at target practice loss, +.5 source replay and +.001 anchor.',
         'src/nowcast_train.py', 'loss=forecast_loss(model(xt[i])'),
     cue('We also check performance', 'Point at {location} ? eligibility and the retained best checkpoint.',
         'src/nowcast_train.py', 'eligible=sv["MSE"]')],
    [cue('This function gives us', 'Switch to {location} ? PeakNowcast.experts(). Point at the five entries in p.',
         'src/peak_nowcast.py', 'def experts'),
     cue('From the earlier', 'Point at {location} ? [:,:1], selecting the earlier model\'s first forecast hour.',
         'src/peak_nowcast.py', 'reference=self.reference.components'),
     cue('We then give each', 'Scroll to {location} ? forward(). Point at the selected weights and weighted sum.',
         'src/peak_nowcast.py', 'def forward(self,x)'),
     cue('Which shares we use', 'Jump to {location} ? observed_regime(). Point at current, previous and acceleration.',
         'src/peak_nowcast.py', 'def observed_regime')],
    [cue('This function makes them', 'Switch to {location} ? collect_episodes(). Point at input_windows, the 24-hour loop, and the three returned arrays.',
         'src/peak_nowcast_train.py', 'def collect_episodes'),
     cue('Here, we calculate', 'Scroll to {location} ? design(). Point at the peak/trough weights, ramp differences and event loss.',
         'src/peak_nowcast_train.py', 'def design'),
     cue('We then choose shares', 'Scroll to {location} ? fit_weights() objective. Point at errors on source examples and target history, plus the penalty for changing the original shares.',
         'src/peak_nowcast_train.py', 'objective=lambda'),
     cue('The solver shown here', 'Point at {location} ? minimize(..., method="SLSQP"). The individual models stay the same; only their shares change.',
         'src/peak_nowcast_train.py', 'fit=minimize(objective'),
     cue('Finally, we check', 'Point at {location} ? the two error constraints. Then show the sparse-regime bounds just above and acceptance/fallback below.',
         'src/peak_nowcast_train.py', 'constraints=constraints+[{"type":"ineq","fun":lambda v:1.1')],
    [cue('This function loads', 'Switch to {location} ? load_selected(). Point at how the chosen dataset and prediction length select a model.',
         'src/submission.py', 'def load_selected'),
     cue('First, we check', 'Jump to {location} ? read_input(). Point at CSV and hourly-timestamp validation.',
         'src/submission.py', 'def read_input'),
     cue('Then we take', 'Scroll to {location} ? predict_window(). Point at the input slice, forecast call, then separate truth slice.',
         'src/submission.py', 'def predict_window'),
     cue('This option in the app', 'Switch to {location} ? main(). Point at the Prediction type radio and its horizon options.',
         'app/submission_app.py', 'horizon = st.radio')],
    [cue('Finally, we check', 'Switch to {location} ? all_metrics(). Point at MSE and MAE across all predicted columns.',
         'src/metrics.py', 'def all_metrics'),
     cue('For temperature RMSE,', 'Stay in all_metrics(); point at putting values back in their original units, and key_RMSE.'),
     cue('We also check errors', 'Switch to {location} ? peak_metrics(). Point at actual peak/trough indices and errors.',
         'src/diagnose_peaks.py', 'def peak_metrics'),
     cue('The saved test log', 'Switch to {location} ? the final 47-tests / OK summary; do not rerun tests.',
         'results/submission/validation.log', 'Ran 47 tests'),
     cue('The app uses one', 'Keep the saved log visible. Explain benchmark scope, then finish without another file switch.')],
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

Open `CODE_WALKTHROUGH.html` and `DASHBOARD_DEMO.html` in a browser. Both work offline and include a rehearsal clock, speaker timeline, screen actions, spoken text and printable views. The spoken paragraphs use plain language. Technical function names stay in the silent screen cues. The code script also includes verified line references and excerpts, with bracketed file/function cues directly between the spoken lines. Follow the cues silently; do not read them aloud.

- Code: Rishik 0:00-4:35; Ayush 4:35-9:15. Leave 45 seconds within the maximum ten minutes.
- Demo: Ayush 0:00-2:30; Rishik 2:30-4:45. Reserve 15 seconds for transitions in the five-minute video.

Use both faces in the code recording, readable screen text and clear audio. Keep these scripts on another device or non-shared screen. Rehearse the real navigation. Do not run training or download data during recording. `source_map.json` contains the script text and exact source anchors; regenerate after source changes with `python -m src.build_recording_guides`.

The app is launched from the repository root with `python -m streamlit run app/app.py`. In the extracted complete package, `python START_APP.py` also works. Use the URL printed by the server. Initial demo state: climate, Next hour, Peak-calibrated hybrid, Domain 1 source, bundled sample, input-only checkbox off. Upload the Delhi sample after switching to the target domain.
''',encoding='utf-8')


if __name__ == '__main__':
    main()
