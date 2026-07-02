import argparse

from src.gee_utils import initialize_earth_engine


def main():
    parser = argparse.ArgumentParser(description="Test Earth Engine initialization.")
    parser.add_argument('--gee-project', type=str, default=None, help='Google Cloud project ID for Earth Engine initialization.')
    parser.add_argument('--service-account-file', type=str, default=None, help='Path to service account JSON file for Earth Engine authentication.')
    parser.add_argument('--force-service-account', action='store_true', help='Require service account auth and fail if the file is missing.')
    args = parser.parse_args()

    initialize_earth_engine(
        project=args.gee_project,
        service_account_file=args.service_account_file,
        force_service_account=args.force_service_account
    )
    print("Earth Engine Connected Successfully!")


if __name__ == '__main__':
    main()