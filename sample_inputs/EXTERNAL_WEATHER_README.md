# New-city weather upload check: Pune

Use these files to test the CSV interface with a third city and a period outside
the original 2021–2024 Shimla/Delhi weather benchmark. These are downloaded
historical gridded weather values, not synthetic examples or station measurements.

## Files

- `weather_pune_2025-10-01_2025-10-31.csv`: 744 consecutive hourly rows for
  1–31 October 2025. Includes enough history and subsequent actuals for either
  next-hour or 24-hour forecast evaluation.
- `weather_pune_2025-10-01_2025-10-31_history_only.csv`: the final 168 hours
  (25–31 October). Use this to predict after the last row, without actual future
  values or an error score. Do not use the full file to claim actuals exist beyond
  31 October: both files end at the same timestamp.

## Try it

1. Open the running app, normally `http://localhost:8502` in this session.
2. Select **Climate** and **Domain 2 · target**. This chooses the existing
   Delhi-adapted checkpoint/scaler; the Delhi label describes that checkpoint.
   The uploaded data itself is Pune, not Delhi. Uploading does not automatically
   train or adapt a new Pune-specific model.
3. Upload the full CSV using **Upload CSV**. Select **Next hour** with
   **Peak-calibrated hybrid**, or **Next 24 hours** with **Strict forecasting hybrid**.
4. Keep **Forecast after the final uploaded row** off to compare forecasts with
   known actuals. Change the origin or use **Run historical evaluation** for a
   broader view. The hourly mode can evaluate up to 576 origins; the 24-hour mode
   can evaluate 24 separate origins. Each uses only preceding observations.
5. Upload the history-only CSV to test prediction beyond the file. The app should
   automatically enable **Forecast after the final uploaded row**. Expect one or
   24 future timestamps and no measured performance until actuals are available.

Both models forecast all seven variables. Temperature is the default plot.
New-city scores can differ from the Shimla/Delhi benchmark; a valid upload does
not guarantee good accuracy on a new distribution.

## Schema and provenance

Columns: `date`, `temperature_2m` (°C), `relative_humidity_2m` (%), `dew_point_2m`
(°C), `surface_pressure` (hPa), `wind_speed_10m` (km/h), `cloud_cover` (%), and
`shortwave_radiation` (W/m²). Times are consecutive hourly **Asia/Kolkata local
wall-clock times**. No values were filled or interpolated.

Provider: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
using its default best-match historical product. Requested coordinates:
18.5204° N, 73.8567° E. The untouched response and exact request, returned grid
coordinates, units, retrieval time and file hashes are preserved under
`data/raw/external_weather/`. The original benchmark data and models are unchanged.

Recreate the download from the project root:

```powershell
python -m src.make_external_weather_sample
```

Upload and forecast checks for both files/horizons are recorded in
`results/external_weather/pune_upload_check.json`. This checks a new input file
through the saved model and does not add Pune to training or benchmark tables.
