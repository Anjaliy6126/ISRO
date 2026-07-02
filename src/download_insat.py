import os
import argparse
import requests
from urllib.parse import urlparse


def make_filename_from_url(url):
    parsed = urlparse(url)
    return os.path.basename(parsed.path)


def download_file(url, out_dir, skip_existing=False):
    os.makedirs(out_dir, exist_ok=True)
    filename = make_filename_from_url(url)
    if not filename:
        raise ValueError(f"Cannot infer filename from URL: {url}")
    out_path = os.path.join(out_dir, filename)
    if skip_existing and os.path.exists(out_path):
        print(f"Skipping existing file: {out_path}")
        return out_path

    print(f"Downloading {url} -> {out_path}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return out_path


def load_urls(urls_file):
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return urls


def generate_urls(template, start_date, end_date):
    urls = []
    current = start_date
    while current <= end_date:
        urls.append(template.format(date=current.strftime("%Y_%m_%d"), date_iso=current.strftime("%Y-%m-%d")))
        current += timedelta(days=1)
    return urls


def main():
    parser = argparse.ArgumentParser(description="Download INSAT-3D AOD files from MOSDAC or a provided URL list.")
    parser.add_argument("--out-dir", type=str, default="data/raw/insat3d", help="Target directory for INSAT-3D files.")
    parser.add_argument("--url", type=str, help="Single file URL to download.")
    parser.add_argument("--urls-file", type=str, help="Path to a text file containing one URL per line.")
    parser.add_argument("--template", type=str, help="URL template with {date} or {date_iso} placeholders.")
    parser.add_argument("--start-date", type=str, help="Start date for template-based download (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="End date for template-based download (YYYY-MM-DD).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist locally.")
    args = parser.parse_args()

    if args.url:
        urls = [args.url]
    elif args.urls_file:
        urls = load_urls(args.urls_file)
    elif args.template:
        if not args.start_date or not args.end_date:
            raise ValueError("When using --template, both --start-date and --end-date are required.")
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        urls = generate_urls(args.template, start_date, end_date)
    else:
        raise ValueError("Provide one of --url, --urls-file, or --template.")

    for url in urls:
        download_file(url, args.out_dir, skip_existing=args.skip_existing)

    print("INSAT-3D download completed.")


if __name__ == "__main__":
    from datetime import datetime, timedelta
    main()
