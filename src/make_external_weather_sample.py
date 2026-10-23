"""Download an external-city upload example, without changing benchmark data.

Defaults to Pune in October 2025: a new city and a period after the weather
benchmark. Saves the untouched API response, request provenance and two CSVs.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .config import DATASETS, RAW_DIR, SAMPLE_DIR
from .submission import read_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', default='Pune')
    parser.add_argument('--latitude', type=float, default=18.5204)
    parser.add_argument('--longitude', type=float, default=73.8567)
    parser.add_argument('--start', default='2025-10-01')
    parser.add_argument('--end', default='2025-10-31')
    args = parser.parse_args()
    slug = ''.join(c.lower() if c.isalnum() else '_' for c in args.city).strip('_')
    if not slug:
        raise ValueError('Provide a city name containing letters or numbers.')
    stem = f'weather_{slug}_{args.start}_{args.end}'
    columns = DATASETS['climate']['columns']
    params = dict(latitude=args.latitude, longitude=args.longitude,
                  start_date=args.start, end_date=args.end,
                  hourly=','.join(columns), timezone='Asia/Kolkata',
                  temperature_unit='celsius', wind_speed_unit='kmh')
    response = requests.get('https://archive-api.open-meteo.com/v1/archive',
                            params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    frame = pd.DataFrame(payload['hourly']).rename(columns={'time': 'date'})
    frame = frame[['date', *columns]]
    frame['date'] = pd.to_datetime(frame['date'])
    expected = pd.date_range(args.start, pd.Timestamp(args.end) + pd.Timedelta(hours=23), freq='h')
    if not pd.DatetimeIndex(frame['date']).equals(expected):
        raise ValueError('API response is not the complete requested hourly period.')
    if len(frame) < 192 or not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError('Need at least 192 hourly rows with no missing/nonfinite values.')
    units = payload['hourly_units']
    for field, expected_unit in [('temperature_2m', '\u00b0C'), ('dew_point_2m', '\u00b0C'),
                                 ('relative_humidity_2m', '%'), ('surface_pressure', 'hPa'),
                                 ('wind_speed_10m', 'km/h'), ('cloud_cover', '%'),
                                 ('shortwave_radiation', 'W/m\u00b2')]:
        if units[field] != expected_unit:
            raise ValueError(f'Unexpected unit for {field}: {units[field]}')
    dest = RAW_DIR/'external_weather'
    dest.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(exist_ok=True)
    raw_path = dest/f'{stem}.json'
    raw_path.write_bytes(response.content)
    full = SAMPLE_DIR/f'{stem}.csv'
    history = SAMPLE_DIR/f'{stem}_history_only.csv'
    frame.to_csv(full, index=False, date_format='%Y-%m-%d %H:%M:%S')
    # This file has just enough lookback, and deliberately no actual future rows.
    frame.tail(168).to_csv(history, index=False, date_format='%Y-%m-%d %H:%M:%S')
    for path in (full, history):
        read_input(path, 'climate')
    provenance = dict(city=args.city, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                      request_url=response.url, request_parameters=params,
                      provider='Open-Meteo Historical Weather API',
                      documentation='https://open-meteo.com/en/docs/historical-weather-api',
                      product='API default best match historical gridded weather; not a station sensor export',
                      returned_grid={k: payload.get(k) for k in
                                     ['latitude', 'longitude', 'elevation', 'timezone', 'utc_offset_seconds']},
                      units=units, timezone_convention='Asia/Kolkata local wall-clock timestamps, without UTC offset',
                      missing_values_filled=False, rows=len(frame),
                      files={str(p.relative_to(SAMPLE_DIR.parent)).replace('\\', '/'):
                             hashlib.sha256(p.read_bytes()).hexdigest() for p in (full, history, raw_path)},
                      usage='External upload smoke check only. No model training, scaler fitting or benchmark changes.')
    (dest/f'{stem}_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(full)
    print(history)
    print(f'{len(frame)} consecutive hourly observations; no missing values; app CSV reader passed.')


if __name__ == '__main__':
    main()
