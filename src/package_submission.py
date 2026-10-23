"""Stage a self-contained local handoff; never publish or push it remotely."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from .config import ROOT


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    name='DeepGen_IE643_CourseProject_IntensiveAssessment'
    parent=ROOT/'submission';stage=parent/name
    stage.mkdir(parents=True,exist_ok=True)
    assert stage.resolve().is_relative_to(parent.resolve())
    def copy(p,relative=None):
        dest=stage/(relative or p.relative_to(ROOT));dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    # Preserve all model families and experiment provenance, including negative controls.
    for folder in ('app','src','tests','data','models','results','.streamlit','recording'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'):
                copy(p)
    for p in (ROOT/'report'/'intensive').rglob('*'):
        if p.is_file():copy(p)
    for p in (ROOT/'report'/'official_template').glob('*'):
        if p.is_file():copy(p)
    for filename in ('IE643_ASSESSMENT_GUIDE.html','IE643_ASSESSMENT_GUIDE.md','assessment_guide.template.md','team_details.json',
                     'IE643_Forecast_Horizons_Results.html','IE643_Strict_Refinement_Report.html','IE643_LiteTCN_Model_Results.html',
                     'IE643_Progressive_LiteTCN_Results.html','IE643_Coupled_LiteTCN_Results.html'):
        copy(ROOT/'report'/filename)
    for p in (ROOT/'presentation'/'intensive').glob('*'):
        if p.is_file():copy(p)
    for filename in ('build_intensive.js','package.json','IE643_Prep_Presentation.pptx'):
        copy(ROOT/'presentation'/filename)
    for p in (ROOT/'presentation'/'assets').glob('*.png'):
        copy(p)
    for p in (ROOT/'dashboard').glob('*.html'):copy(p)
    for p in (ROOT/'sample_inputs').glob('*.csv'):copy(p)
    for p in (ROOT/'sample_inputs').glob('*.md'):copy(p)
    for filename in ('requirements.txt','requirements-experiments.txt','requirements-documentation.txt'):
        copy(ROOT/filename)
    copy(ROOT/'README.md',Path('PROJECT_README.md'))
    copy(ROOT/'report/intensive/IE643_Intensive_Assessment_Report.pdf',Path('Report.pdf'))
    copy(ROOT/'presentation/intensive/IE643_Intensive_Assessment.pptx',Path('Presentation.pptx'))
    copy(ROOT/'dashboard/IE643_Results_Dashboard.html',Path('Dashboard.html'))
    (stage/'START_APP.py').write_text('''"""Run from an extracted copy of the complete package."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
result=subprocess.run([sys.executable,'-m','streamlit','run',str(root/'app'/'app.py'),'--server.address','127.0.0.1'],cwd=root)
raise SystemExit(result.returncode)
''',encoding='utf-8')
    (stage/'VERIFY_PACKAGE.py').write_text('''"""Verify every staged file against its SHA-256 manifest."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
bad=[]
for name,expected in manifest['files'].items():
 p=root/name
 if not p.is_file():bad.append(name+' (missing)');continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 if h.hexdigest()!=expected:bad.append(name+' (changed)')
if bad:raise SystemExit('Verification failed: '+repr(bad))
print('Verified',len(manifest['files']),'files. No model, data or evidence-file hashes changed.')
''',encoding='utf-8')
    (stage/'README.md').write_text('''# DeepGen - IE643 intensive assessment handoff

Ayush Patel (23B0037) and Rishik Yadav (23B2103).

## Start here

- **Report.pdf**: new technical report compiled from the supplied course template; complete LaTeX, numerical tables and bibliography in `report/intensive/`.
- **Presentation.pptx**: 18 content slides plus two reference slides; practice notes in `presentation/intensive/IE643_Intensive_Speaker_Notes.md`.
- **Dashboard.html**: interactive offline results viewer. Open directly in a browser; all charts and Plotly are embedded.
- **START_APP.py**: CSV forecasting interface for all three datasets and both domains. This requires Python dependencies; the HTML dashboard does not.
- **report/IE643_ASSESSMENT_GUIDE.html**: complete project study guide and viva preparation.
- **recording/CODE_WALKTHROUGH.html** and **recording/DASHBOARD_DEMO.html**: timed two-person recording scripts with exact screen actions and a rehearsal clock.

## Launch the CSV app

Extract the **entire** ZIP first. Keep `app`, `src`, `models`, `results`, `sample_inputs` and the other folders together. Tested with Python 3.13.2 and CPU Torch 2.6.0 on Windows.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-experiments.txt
.venv/Scripts/python START_APP.py
```

The terminal prints the local browser URL. Select a dataset, horizon and source/target domain. Use a sample or upload one chronological CSV. Hourly data needs consecutive timestamps and all seven variables. Finance accepts processed features or raw OHLCV. With only a complete lookback, forecast-after-final-row mode works without an error score. With actual futures, the app compares source/adapted, persistence, seasonal and input-mean forecasts and exports predictions/actuals/metrics. It does not train on uploaded future labels.

For weather and energy, use **Prediction type** at the top of the app to select **Next hour** or **Next 24 hours**. This switches the checkpoint horizon, output timestamps, plot, evaluation and saved benchmark results together. The uploaded CSV stays available when you change prediction type. Finance retains its five-trading-day horizon.

Source-only and adapted checkpoints are included. The deployment app uses seed 0; reported benchmark means average seeds 0,1,2. Other model families and negative controls remain available in the separate research explorer: `python -m streamlit run app/research.py`.

## Verify and reproduce

```powershell
python VERIFY_PACKAGE.py
python -m unittest discover -s tests -v
python -m src.check_horizon_models
```

The latest test logs and artifact audit are in `results/submission/` and `results/forecast_horizons/`. The package includes processed/raw data, all saved model families, selection manifests, CSVs, plots and code. Earlier broad-history pilots and labeled diagnostic fits are preserved as historical controls; they are not strict label-free submission claims. Model training is unnecessary for the demo.

Optional documentation rebuilding needs `requirements-documentation.txt`. Generate numerical tables with `python -m src.build_submission_assets`, generate the template source with `python -m src.build_intensive_report`, then compile from `report/intensive/` using `tectonic IE643_Intensive_Assessment_Report.tex` or a compatible LaTeX/BibTeX installation. The supplied template and provenance record are preserved. The PPTX source is `presentation/build_intensive.js`; install its declared Node packages before rebuilding it.

The configured GitHub remote is https://github.com/Ayush20990/ie643-project . The shareable intensive-assessment release is https://github.com/Ayush20990/ie643-project/releases/latest . The repository and release are the collaboration handoff; building this ZIP alone does not publish it. No Drive upload, course submission or partner messaging is performed by the package builder.

## Team contributions supplied by the team

- Both: research of methods/topics and carrying out experiments.
- Ayush: dashboard and application development.
- Rishik: report and presentation preparation.

## Important interpretation

H=1 is rolling next-hour prediction with refreshed observed history. H=24 is a single-origin day-ahead task. Finance H=5 predicts engineered features, not price levels/profits. The final model does not win every metric: energy trough MAE still favors persistence. Development tests were reused. See report for matched-horizon controls and negative results.

## Before actual submission

The supplied assessment instructions require the team's own report account and warn against simply submitting AI-generated report text. Treat the PDF/LaTeX as a verified technical draft: both authors must review the evidence and rewrite the narrative in their own words before submitting it. No claim of course approval or a guaranteed mark is made.

The team must still record the **maximum 10-minute code walkthrough** and **5-minute demo**, with the required participation/face visibility, confirm TA dataset approval, check the actual deadline/slot and access settings, and upload the required materials. No placeholder recordings are included. See `VIDEO_RECORDING_GUIDE.md` for the sequence. The supplied deadline is 25 October 2026, 11:59 PM IST; presentation is 12 minutes plus 3 minutes Q&A.
''',encoding='utf-8')
    (stage/'VIDEO_RECORDING_GUIDE.md').write_text('# DeepGen recording guide\n\nOpen these offline HTML scripts in a browser:\n\n- [Code walkthrough](recording/CODE_WALKTHROUGH.html): Rishik 0:00-4:35, Ayush 4:35-9:15; leave 45 seconds within the maximum ten minutes.\n- [Dashboard demo](recording/DASHBOARD_DEMO.html): Ayush 0:00-2:30, Rishik 2:30-4:45; reserve 15 seconds for transitions in the five-minute video.\n\nBoth guides include spoken paragraphs, exact screen actions, a speaker timeline, rehearsal clock and printable view. The code guide includes verified source excerpts. Keep the guide on another device or an unshared screen; rehearse navigation before recording. Both members participate and keep faces visible in the code recording. These are scripts, not completed recordings.\n\nLaunch the app with `python START_APP.py` after extracting the complete ZIP and installing dependencies. Start the demo with climate, Next hour, Peak-calibrated hybrid, Domain 1 source, the bundled sample and the input-only checkbox off. Switch to Domain 2 before uploading `sample_inputs/climate_domain2_delhi.csv`. Do not train models or download datasets during recording. The scripts cover both forecast horizons, energy, the weak finance results and downloads.\n\nFor the live presentation, use the separate deck notes and its shorter demonstration.\n',encoding='utf-8')
    status={'team':'DeepGen','technical_artifacts':'built locally','report_author_narrative_review':'required under supplied assessment instructions',
            'recordings':'not recorded; team faces/participation required','TA_dataset_approval':'not supplied','remote_push_or_upload':'GitHub handoff published; see https://github.com/Ayush20990/ie643-project/releases/latest. Local rebuilds require separate publication.',
            'check_logs':['results/submission/validation.log','results/submission/presentation_validation.json','results/submission/document_validation.json']}
    (stage/'HANDOFF_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    files={str(p.relative_to(stage)).replace('\\','/'):digest(p) for p in sorted(stage.rglob('*')) if p.is_file() and p.name!='PACKAGE_MANIFEST.json'}
    (stage/'PACKAGE_MANIFEST.json').write_text(json.dumps({'package':name,'files':files},indent=2),encoding='utf-8')
    archive=parent/(name+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,str(Path(name)/p.relative_to(stage)))
    print(stage);print(archive);print('Files:',len(files),'ZIP MiB:',round(archive.stat().st_size/1024**2,1))


if __name__=='__main__':main()
