import os
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta


def load_api_key(provided_key):
    return provided_key or os.environ.get('FIRMS_API_KEY')


def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def download_fire_day(api_key, date_str, out_path, bbox='68,8,98,38', source='MODIS_SP'):
    url = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{bbox}/1/{date_str}'
    print(f'Downloading FIRMS active fire data for {date_str}')
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

    return out_path


def combine_daily_files(file_list, target_path):
    dfs = []
    for path in file_list:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as exc:
            print(f'Warning: could not read {path}: {exc}')
    if len(dfs) == 0:
        raise RuntimeError('No fire data files were downloaded.')
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(target_path, index=False)
    return combined


def main():
    parser = argparse.ArgumentParser(description='Download MODIS/VIIRS fire hotspot data from NASA FIRMS.')
    parser.add_argument('--start-date', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default=None, help='End date YYYY-MM-DD. Defaults to start date.')
    parser.add_argument('--api-key', type=str, default=None, help='NASA FIRMS API key. Can also be provided via FIRMS_API_KEY environment variable.')
    parser.add_argument('--out-dir', type=str, default='data/raw', help='Directory to store fire_alerts.csv')
    parser.add_argument('--source', type=str, default='MODIS_SP', choices=['MODIS_SP', 'VIIRS_SNPP_SP', 'VIIRS_NOAA20_SP'], help='FIRMS source collection.')
    parser.add_argument('--bbox', type=str, default='68,8,98,38', help='Bounding box min_lon,min_lat,max_lon,max_lat.')
    parser.add_argument('--skip-existing', action='store_true', help='Skip download if the output file exists.')
    args = parser.parse_args()

    api_key = load_api_key(args.api_key)
    if not api_key:
        raise ValueError('FIRMS API key is required via --api-key or FIRMS_API_KEY environment variable.')

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else start_date

    daily_files = []
    for current in date_range(start_date, end_date):
        date_str = current.strftime('%Y-%m-%d')
        daily_path = os.path.join(args.out_dir, f'fire_alerts_{date_str}.csv')
        if args.skip-existing and os.path.exists(daily_path):
            print(f'Skipping existing daily fire file: {daily_path}')
        else:
            download_fire_day(api_key, date_str, daily_path, bbox=args.bbox, source=args.source)
        daily_files.append(daily_path)

    combined_path = os.path.join(args.out_dir, 'fire_alerts.csv')
    combine_daily_files(daily_files, combined_path)
    print(f'Combined fire alerts saved to {combined_path}')


if __name__ == '__main__':
    main()
