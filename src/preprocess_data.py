import os
import pandas as pd
from datetime import datetime


def clean_cpcb_stations(raw_path, output_path=None):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"CPCB station file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df.columns = [col.strip() for col in df.columns]

    common_renames = {
        'Station Code': 'station_id',
        'Station Id': 'station_id',
        'Station': 'station_id',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Lat': 'latitude',
        'Lon': 'longitude',
        'Date': 'date',
        'DATE': 'date',
        'Datetime': 'date',
        'DateTime': 'date',
        'Time': 'time',
        'PM2.5': 'PM2.5',
        'PM10': 'PM10',
        'SO2': 'SO2',
        'NO2': 'NO2',
        'CO': 'CO',
        'O3': 'O3',
        'AOD': 'AOD'
    }
    df = df.rename(columns={k: v for k, v in common_renames.items() if k in df.columns})

    if 'date' not in df.columns:
        raise ValueError('Could not find a date column in CPCB station data.')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'time' in df.columns:
        df['time'] = df['time'].astype(str).str.strip()
        df.loc[df['time'].notna(), 'date'] = pd.to_datetime(
            df.loc[df['time'].notna(), 'date'].dt.strftime('%Y-%m-%d') + ' ' + df.loc[df['time'].notna(), 'time'],
            errors='coerce'
        )

    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError('CPCB station file must contain latitude and longitude fields.')

    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    df = df.dropna(subset=['date', 'latitude', 'longitude'])
    if 'station_id' not in df.columns:
        df['station_id'] = df.index.astype(str)

    # Standardize pollutant columns if they exist
    pollutant_cols = [col for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'] if col in df.columns]
    for col in pollutant_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if output_path is None:
        output_path = raw_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned CPCB station data to {output_path}. Rows: {len(df)}")
    return output_path


def clean_fire_alerts(raw_path, output_path=None):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Fire alerts file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df.columns = [col.strip() for col in df.columns]

    if 'acq_date' in df.columns:
        df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
    elif 'date' in df.columns:
        df['acq_date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        raise ValueError('Fire alerts data must contain acq_date or date field.')

    latitude_cols = [c for c in ['latitude', 'Latitude', 'LATITUDE'] if c in df.columns]
    longitude_cols = [c for c in ['longitude', 'Longitude', 'LONGITUDE'] if c in df.columns]
    if not latitude_cols or not longitude_cols:
        raise ValueError('Fire alerts data must contain latitude and longitude fields.')

    df['latitude'] = pd.to_numeric(df[latitude_cols[0]], errors='coerce')
    df['longitude'] = pd.to_numeric(df[longitude_cols[0]], errors='coerce')

    for numeric_col in ['frp', 'confidence', 'brightness', 'scan', 'track']:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')

    df = df.dropna(subset=['acq_date', 'latitude', 'longitude'])
    if output_path is None:
        output_path = raw_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned fire alert data to {output_path}. Rows: {len(df)}")
    return output_path


def combine_fire_alerts(input_dir, output_file):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Fire alerts directory not found: {input_dir}")

    files = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.lower().endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames = []
    for file_path in files:
        try:
            frames.append(pd.read_csv(file_path))
        except Exception as exc:
            print(f"Warning: unable to read {file_path}: {exc}")

    if not frames:
        raise RuntimeError("No fire alert files could be combined.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"Combined {len(frames)} fire alert files into {output_file}. Rows: {len(combined)}")
    return output_file


def main(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    raw_dir = os.path.join(base_dir, 'data', 'raw')
    cpcb_raw = os.path.join(raw_dir, 'cpcb_stations.csv')
    fire_raw = os.path.join(raw_dir, 'fire_alerts.csv')
    fire_raw_dir = os.path.join(raw_dir, 'fire_alerts')

    if os.path.exists(cpcb_raw):
        clean_cpcb_stations(cpcb_raw, cpcb_raw)
    else:
        print(f"Skipping CPCB preprocessing because {cpcb_raw} is missing.")

    if os.path.exists(fire_raw_dir):
        combine_fire_alerts(fire_raw_dir, fire_raw)

    if os.path.exists(fire_raw):
        clean_fire_alerts(fire_raw, fire_raw)
    else:
        print(f"Skipping fire alert preprocessing because {fire_raw} is missing.")

    return {
        'cpcb': cpcb_raw if os.path.exists(cpcb_raw) else None,
        'fire_alerts': fire_raw if os.path.exists(fire_raw) else None
    }

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess raw CPCB and fire alert datasets for the pipeline.')
    parser.add_argument('--base-dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
                        help='Root project directory. Defaults to repo root.')
    args = parser.parse_args()
    main(base_dir=args.base_dir)
