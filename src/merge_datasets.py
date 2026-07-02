import os
from datetime import datetime, timedelta
import xarray as xr
from src.preprocessing import load_daily_netcdf


def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def build_daily_dataset(date_str, raw_dir):
    ds = load_daily_netcdf(date_str, raw_dir)
    if ds is None:
        return None
    ds = ds.expand_dims(time=[datetime.strptime(date_str, "%Y-%m-%d")])
    return ds


def merge_daily_grids(start_date, end_date, raw_dir, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    datasets = []

    for current in date_range(start_date, end_date):
        date_str = current.strftime("%Y-%m-%d")
        ds = build_daily_dataset(date_str, raw_dir)
        if ds is not None:
            datasets.append(ds)
        else:
            print(f"Warning: daily dataset missing for {date_str}")

    if not datasets:
        raise RuntimeError("No daily grid datasets found for the requested date range.")

    merged = xr.concat(datasets, dim="time")
    merged.to_netcdf(output_path)

    print(f"Merged daily gridded dataset saved to {output_path}."
          f" Days included: {len(datasets)}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge daily raw grids into a single processed NetCDF file.")
    parser.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD. Defaults to start date.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to the raw data directory.")
    parser.add_argument("--out-file", type=str, default="data/processed/merged_daily_grids.nc", help="Target merged output file.")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else start_date
    merge_daily_grids(start_date, end_date, args.raw_dir, args.out_file)
