import os
import argparse
from datetime import datetime, timedelta
import cdsapi


def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def download_era5_day(date_str, out_dir, variables=None, area=None):
    if variables is None:
        variables = [
            '2m_temperature',
            '2m_dewpoint_temperature',
            'boundary_layer_height',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation',
            'surface_pressure'
        ]
    if area is None:
        area = [38.0, 68.0, 8.0, 98.0]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'ERA5_daily_{date_str.replace("-", "_")}.nc')
    print(f'Downloading ERA5 for {date_str} -> {out_path}')
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variables,
            'year': date_str[:4],
            'month': date_str[5:7],
            'day': date_str[8:10],
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00',
                '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00',
                '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00'
            ],
            'area': area,
        },
        out_path
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Download ERA5 meteorological reanalysis data for India using CDS API.')
    parser.add_argument('--start-date', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, default=None, help='End date YYYY-MM-DD. Defaults to start date.')
    parser.add_argument('--out-dir', type=str, default='data/raw/era5', help='Output directory for ERA5 files.')
    parser.add_argument('--skip-existing', action='store_true', help='Skip files that already exist.')
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else start_date

    for current in date_range(start_date, end_date):
        date_str = current.strftime('%Y-%m-%d')
        out_path = os.path.join(args.out_dir, f'ERA5_daily_{date_str.replace("-", "_")}.nc')
        if args.skip-existing and os.path.exists(out_path):
            print(f'Skipping existing ERA5 file: {out_path}')
            continue
        download_era5_day(date_str, args.out_dir)

    print('ERA5 download completed.')


if __name__ == '__main__':
    main()
