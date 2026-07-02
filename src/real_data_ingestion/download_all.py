import os
import sys
import argparse
import subprocess
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SCRIPT_PATHS = {
    "cpcb": os.path.join(BASE_DIR, "src", "download_cpcb.py"),
    "firms": os.path.join(BASE_DIR, "src", "download_firms.py"),
    "era5": os.path.join(BASE_DIR, "src", "download_meteorology.py"),
    "s5p": os.path.join(BASE_DIR, "src", "download_sentinel5p.py"),
    "mosdac": os.path.join(BASE_DIR, "src", "real_data_ingestion", "download_mosdac.py"),
    "preprocess": os.path.join(BASE_DIR, "src", "preprocess_data.py"),
    "merge": os.path.join(BASE_DIR, "src", "merge_datasets.py")
}


def run_python_script(script_path, args):
    cmd = [sys.executable, script_path] + args
    print("\nRunning:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")


def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def download_cpcb(url, out_file, skip_existing):
    if skip_existing and os.path.exists(out_file):
        print(f"Skipping existing CPCB file: {out_file}")
        return
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    run_python_script(SCRIPT_PATHS["cpcb"], ["--url", url, "--out-file", out_file, "--skip-existing"] if skip_existing else ["--url", url, "--out-file", out_file])


def download_firms(api_key, start_date, end_date, out_dir, skip_existing):
    args = ["--start-date", start_date, "--end-date", end_date, "--api-key", api_key, "--out-dir", out_dir]
    if skip_existing:
        args.append("--skip-existing")
    run_python_script(SCRIPT_PATHS["firms"], args)


def download_era5(start_date, end_date, out_dir, skip_existing):
    args = ["--start-date", start_date, "--end-date", end_date, "--out-dir", out_dir]
    if skip_existing:
        args.append("--skip-existing")
    run_python_script(SCRIPT_PATHS["era5"], args)


def download_s5p(start_date, end_date, out_dir, skip_existing, gee_project=None, service_account_file=None):
    args = ["--start-date", start_date, "--end-date", end_date, "--out-dir", out_dir]
    if skip_existing:
        args.append("--skip-existing")
    if gee_project:
        args.extend(["--gee-project", gee_project])
    if service_account_file:
        args.extend(["--service-account-file", service_account_file])
    run_python_script(SCRIPT_PATHS["s5p"], args)


def convert_mosdac_files(h5_source, out_dir, skip_existing):
    if os.path.isdir(h5_source):
        for filename in os.listdir(h5_source):
            if filename.lower().endswith(('.h5', '.hdf5')):
                src_path = os.path.join(h5_source, filename)
                out_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".nc")
                if skip_existing and os.path.exists(out_path):
                    print(f"Skipping existing MOSDAC output: {out_path}")
                    continue
                run_python_script(SCRIPT_PATHS["mosdac"], ["--h5_path", src_path, "--out_path", out_path])
    else:
        filename = os.path.splitext(os.path.basename(h5_source))[0] + ".nc"
        out_path = os.path.join(out_dir, filename)
        if not (skip_existing and os.path.exists(out_path)):
            run_python_script(SCRIPT_PATHS["mosdac"], ["--h5_path", h5_source, "--out_path", out_path])


def preprocess_raw_data(base_dir):
    run_python_script(SCRIPT_PATHS["preprocess"], ["--base-dir", base_dir])


def merge_daily_grids(start_date, end_date, raw_dir, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    run_python_script(SCRIPT_PATHS["merge"], [
        "--start-date", start_date,
        "--end-date", end_date,
        "--raw-dir", raw_dir,
        "--out-file", out_file
    ])


def main():
    parser = argparse.ArgumentParser(description="Download raw datasets for the ISRO air quality project.")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=False, help="End date (YYYY-MM-DD). Defaults to start date.")
    parser.add_argument("--cpcb-url", type=str, default=None, help="Official CPCB station CSV URL.")
    parser.add_argument("--firms-api-key", type=str, default=None, help="NASA FIRMS API key for active fire downloads.")
    parser.add_argument("--mosdac-h5", type=str, default=None, help="Path to local MOSDAC HDF5 files for INSAT conversion.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip download/conversion if output files already exist.")
    parser.add_argument("--gee-project", type=str, default=None,
                        help="Google Cloud project ID for Earth Engine initialization during Sentinel-5P download.")
    parser.add_argument("--service-account-file", type=str, default=None,
                        help="Path to a Google Cloud service account JSON file for Earth Engine authentication.")
    parser.add_argument("--only", nargs="*", choices=["cpcb", "firms", "era5", "s5p", "mosdac"],
                        help="Only execute the selected data ingestion steps.")
    parser.add_argument("--merge-output", type=str, default=None,
                        help="Optional output file for merged daily grids after ingestion.")
    args = parser.parse_args()

    if args.only is None:
        kinds = ["cpcb", "firms", "era5", "s5p", "mosdac"]
    else:
        kinds = args.only

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date) if args.end_date else start_date
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    if "cpcb" in kinds and not args.cpcb_url:
        raise ValueError("--cpcb-url is required when downloading CPCB station data.")

    if "firms" in kinds and not args.firms_api_key:
        raise ValueError("--firms-api-key is required when downloading FIRMS fire alerts.")

    if "mosdac" in kinds and not args.mosdac_h5:
        raise ValueError("--mosdac-h5 is required when converting MOSDAC HDF5 files.")

    if "cpcb" in kinds:
        print("\nDownloading CPCB station monitoring data...")
        download_cpcb(args.cpcb_url, os.path.join(RAW_DIR, "cpcb_stations.csv"), args.skip_existing)

    if "firms" in kinds:
        print("\nDownloading NASA FIRMS fire hotspot data...")
        download_firms(args.firms_api_key, start_date_str, end_date_str, os.path.join(RAW_DIR, "fire_alerts"), args.skip_existing)

    if "era5" in kinds:
        print("\nDownloading ERA5 meteorological data...")
        download_era5(start_date_str, end_date_str, os.path.join(RAW_DIR, "era5"), args.skip_existing)

    if "s5p" in kinds:
        print("\nDownloading Sentinel-5P gridded pollutant data...")
        download_s5p(
            start_date_str,
            end_date_str,
            os.path.join(RAW_DIR, "s5p"),
            args.skip_existing,
            gee_project=args.gee_project,
            service_account_file=args.service_account_file
        )

    if "mosdac" in kinds:
        print("\nConverting MOSDAC INSAT-3D HDF5 files to NetCDF...")
        convert_mosdac_files(args.mosdac_h5, os.path.join(RAW_DIR, "insat3d"), args.skip_existing)

    if args.merge_output:
        print(f"\nMerging daily gridded raw datasets into {args.merge_output}...")
        merge_daily_grids(start_date_str, end_date_str, os.path.join(RAW_DIR), args.merge_output)

    print("\nRaw dataset ingestion complete.")
    print("Preprocessing CPCB and fire alert source files...")
    preprocess_raw_data(BASE_DIR)


if __name__ == "__main__":
    main()
