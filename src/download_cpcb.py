import os
import argparse
import requests
import pandas as pd


def download_cpcb_data(url, out_path, skip_existing=False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if skip_existing and os.path.exists(out_path):
        print(f"Skipping existing CPCB file: {out_path}")
        return out_path

    if not url:
        raise ValueError(
            "CPCB data download requires --url. "
            "If an official CPCB API or repository URL is unavailable, download the station CSV manually and save it to data/raw/cpcb_stations.csv."
        )

    print(f"Downloading CPCB station data from {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    # Validate the file can be parsed
    try:
        pd.read_csv(out_path, nrows=5)
    except Exception as exc:
        raise RuntimeError(f"Downloaded CPCB CSV could not be parsed: {exc}")

    print(f"Saved CPCB station CSV to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download CPCB station monitoring data to data/raw.")
    parser.add_argument("--url", type=str, required=True, help="Official CPCB CSV URL or a hosted raw file link.")
    parser.add_argument("--out-file", type=str, default="data/raw/cpcb_stations.csv", help="Target path for the downloaded CPCB CSV.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip download if the target file already exists.")
    args = parser.parse_args()

    download_cpcb_data(args.url, args.out_file, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
