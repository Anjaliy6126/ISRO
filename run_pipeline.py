import os
import sys
import argparse
import subprocess

# Append current directory to path to handle imports properly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_simulator import run_simulation
from src.preprocessing import align_and_match
from src.ml_models import train_and_evaluate_ml
from src.dl_models import train_and_evaluate_dl
from src.hotspot_detection import analyze_fire_hcho_lag_correlation


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DOWNLOAD_SCRIPT = os.path.join(BASE_DIR, "src", "real_data_ingestion", "download_all.py")
PREPROCESS_SCRIPT = os.path.join(BASE_DIR, "src", "preprocess_data.py")


def run_python_script(script_path, args):
    cmd = [sys.executable, script_path] + args
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def print_download_summary(args):
    print("\n=== Real-mode download configuration ===")
    print(f"CPCB URL: {args.cpcb_url or 'not provided'}")
    print(f"FIRMS API key: {'provided' if args.firms_api_key else 'not provided'}")
    print(f"MOSDAC HDF5 path: {args.mosdac_h5 or 'not provided'}")
    print(f"Earth Engine project: {args.gee_project or 'not provided'}")
    print(f"Service account file: {args.service_account_file or 'not provided'}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Skip existing files: {'yes' if args.skip_existing else 'no'}")
    if args.only:
        print(f"Selected ingestion steps: {', '.join(args.only)}")
    else:
        print("Selected ingestion steps: all")
    if args.merge_output:
        print(f"Merge output file: {args.merge_output}")
    print("=======================================\n")


def check_raw_data_exists(base_dir):
    raw_dir = os.path.join(base_dir, "data", "raw")
    missing = []

    cpcb_path = os.path.join(raw_dir, "cpcb_stations.csv")
    fire_path = os.path.join(raw_dir, "fire_alerts.csv")
    insat_dir = os.path.join(raw_dir, "insat3d")
    s5p_dir = os.path.join(raw_dir, "s5p")
    era5_dir = os.path.join(raw_dir, "era5")

    if not os.path.exists(cpcb_path):
        missing.append(cpcb_path)
    if not os.path.exists(fire_path):
        missing.append(fire_path)

    def has_nc_files(directory):
        return os.path.isdir(directory) and any(fname.endswith(".nc") for fname in os.listdir(directory))

    if not has_nc_files(insat_dir):
        missing.append(insat_dir)
    if not has_nc_files(s5p_dir):
        missing.append(s5p_dir)
    if not has_nc_files(era5_dir):
        missing.append(era5_dir)

    return missing


def download_raw_datasets(args):
    if not args.start_date or not args.end_date:
        raise ValueError("--start-date and --end-date are required when --download is enabled.")

    script_args = [
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
    ]

    if args.cpcb_url:
        script_args.extend(["--cpcb-url", args.cpcb_url])

    if args.firms_api_key:
        script_args.extend(["--firms-api-key", args.firms_api_key])

    if args.mosdac_h5:
        script_args.extend(["--mosdac-h5", args.mosdac_h5])

    if args.gee_project:
        script_args.extend(["--gee-project", args.gee_project])

    if args.service_account_file:
        script_args.extend(["--service-account-file", args.service_account_file])

    if args.only:
        script_args.extend(["--only"] + args.only)

    if args.skip_existing:
        script_args.append("--skip-existing")

    if args.merge_output:
        script_args.extend(["--merge-output", args.merge_output])

    run_python_script(DOWNLOAD_SCRIPT, script_args)


def run_preprocessing(base_dir):
    run_python_script(PREPROCESS_SCRIPT, ["--base-dir", base_dir])


def main(use_simulation=True, args=None):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    print("=========================================================")
    print("   AIR QUALITY ESTIMATION & HCHO HOTSPOT DETECTION       ")
    print("                PIPELINE RUNNER                          ")
    print("=========================================================\n")
    
    if use_simulation:
        print("STEP 1/6: GENERATING SYNTHETIC RAW DATA FOR THE PIPELINE")
        run_simulation(base_dir)
    else:
        if args and args.download:
            print_download_summary(args)
            print("STEP 1/6: DOWNLOADING REAL DATA FROM OFFICIAL SOURCES")
            download_raw_datasets(args)
            print("STEP 2/6: PREPROCESSING RAW CPCB AND FIRE ALERT DATA")
            run_preprocessing(base_dir)
        else:
            print("STEP 1/5: USING EXISTING RAW DATA FROM data/raw")
            missing = check_raw_data_exists(base_dir)
            if missing:
                raise FileNotFoundError(
                    "Real-data mode requires existing raw files in data/raw. Missing: " + ", ".join(missing) +
                    ".\nPlease download the real datasets using the ingestion scripts in src/real_data_ingestion/download_all.py or run_pipeline.py --mode real --download."
                )
    print("\n" + "="*50 + "\n")
    
    print("STEP 3/5: RUNNING SPATIOTEMPORAL DATA MATCHING & QUALITY CONTROL")
    align_and_match(base_dir)
    print("\n" + "="*50 + "\n")
    
    print("STEP 4/5: TRAINING AND EVALUATING MACHINE LEARNING REGRESSORS")
    train_and_evaluate_ml(base_dir)
    print("\n" + "="*50 + "\n")
    
    print("STEP 5/5: TRAINING AND EVALUATING PYTORCH CNN-LSTM MODEL")
    train_and_evaluate_dl(base_dir, epochs=15, batch_size=32)
    print("\n" + "="*50 + "\n")
    
    print("STEP 6/6: RUNNING HCHO-FIRE LAG-CORRELATION ANALYSIS")
    analyze_fire_hcho_lag_correlation(base_dir)
    print("\n" + "="*50 + "\n")
    
    print("Pipeline Execution Completed Successfully! All models and dataset outputs are saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ISRO air quality pipeline.")
    parser.add_argument(
        "--mode",
        choices=["simulate", "real"],
        default="simulate",
        help="Run with synthetic data (simulate) or use existing real raw datasets under data/raw (real)."
    )
    parser.add_argument("--download", action="store_true", help="If real mode, download official raw datasets before running the pipeline.")
    parser.add_argument("--start-date", type=str, help="Raw dataset ingestion start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="Raw dataset ingestion end date (YYYY-MM-DD).")
    parser.add_argument("--firms-api-key", type=str, help="NASA FIRMS API key for fire hotspot download.")
    parser.add_argument("--cpcb-url", type=str, help="URL to download CPCB station monitoring CSV data.")
    parser.add_argument("--mosdac-h5", type=str, help="Path to local MOSDAC HDF5 files for INSAT conversion.")
    parser.add_argument("--only", nargs="*", choices=["cpcb", "firms", "era5", "s5p", "mosdac"],
                        help="Only execute the selected ingestion steps.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip download/conversion for files that already exist.")
    parser.add_argument("--gee-project", type=str, default=None,
                        help="Google Cloud project ID for Earth Engine initialization during Sentinel-5P download.")
    parser.add_argument("--service-account-file", type=str, default=None,
                        help="Path to a Google Cloud service account JSON file for Earth Engine authentication.")
    parser.add_argument("--merge-output", type=str, default=None,
                        help="If set, the download wrapper will also merge daily raw grids into the specified NetCDF output file.")
    args = parser.parse_args()
    main(use_simulation=(args.mode == "simulate"), args=args)
