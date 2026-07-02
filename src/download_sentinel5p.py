import os
import argparse
from datetime import datetime, timedelta

import ee
import geemap
import xarray as xr

from src.gee_utils import initialize_earth_engine


def initialize_earth_engine_for_script(project=None, service_account_file=None):
    initialize_earth_engine(project=project, service_account_file=service_account_file)


def build_s5p_image(date_str, region):
    start = ee.Date(date_str)
    end = start.advance(1, 'day')
    collections = {
        'NO2': ('COPERNICUS/S5P/OFFL/L3_NO2', 'NO2_column_number_density'),
        'SO2': ('COPERNICUS/S5P/OFFL/L3_SO2', 'SO2_column_number_density'),
        'CO': ('COPERNICUS/S5P/OFFL/L3_CO', 'CO_column_number_density'),
        'O3': ('COPERNICUS/S5P/OFFL/L3_O3', 'O3_column_number_density'),
        'HCHO': ('COPERNICUS/S5P/OFFL/L3_HCHO', 'tropospheric_HCHO_column_number_density')
    }

    bands = []
    for label, (collection_id, band_name) in collections.items():
        img = ee.ImageCollection(collection_id) \
            .filterDate(start, end) \
            .select(band_name) \
            .mean() \
            .rename(label) \
            .clip(region)
        bands.append(img)

    return ee.Image.cat(bands)


def export_day(date_str, out_dir, scale, region):
    os.makedirs(out_dir, exist_ok=True)
    filename = f"S5P_TROPOMI_daily_{date_str.replace('-', '_')}.tif"
    out_path = os.path.join(out_dir, filename)
    print(f"Exporting Sentinel-5P for {date_str} to {out_path}")

    image = build_s5p_image(date_str, region)
    geemap.ee_export_image(
        image,
        filename=out_path,
        scale=scale,
        region=region.geometry(),
        file_per_band=False,
        crs='EPSG:4326'
    )

    return out_path


def convert_tif_to_netcdf(tif_path, nc_path):
    da = xr.open_rasterio(tif_path)
    ds = da.to_dataset(dim='band')
    band_names = [str(i + 1) for i in range(ds.dims['band'])]
    if 'band' in ds:
        ds = ds.rename({ 'band': 'parameter' })
    ds = ds.assign_coords(parameter=list(['NO2', 'SO2', 'CO', 'O3', 'HCHO']))
    ds = ds.rename({ 'x': 'lon', 'y': 'lat' })
    ds.to_netcdf(nc_path)
    return nc_path


def main():
    parser = argparse.ArgumentParser(description='Download Sentinel-5P daily gridded pollutant products using Google Earth Engine.')
    parser.add_argument('--start-date', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, required=False, help='End date YYYY-MM-DD. Defaults to start date.')
    parser.add_argument('--out-dir', type=str, default='data/raw/s5p', help='Output directory for Sentinel-5P files.')
    parser.add_argument('--scale', type=int, default=10000, help='Spatial resolution in meters for export.')
    parser.add_argument('--skip-existing', action='store_true', help='Skip dates that already have output files.')
    parser.add_argument('--gee-project', type=str, default=None, help='Google Cloud project ID for Earth Engine initialization.')
    parser.add_argument('--service-account-file', type=str, default=None, help='Path to Google Cloud service account JSON file for Earth Engine authentication.')
    parser.add_argument('--force-service-account', action='store_true', help='Require service account auth and fail if the file is missing.')
    args = parser.parse_args()

    initialize_earth_engine_for_script(
        project=args.gee_project,
        service_account_file=args.service_account_file,
        force_service_account=args.force_service_account
    )
    import ee
    india = ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME', 'India'))

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else start_date

    for current in (start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)):
        date_str = current.strftime('%Y-%m-%d')
        nc_path = os.path.join(args.out_dir, f"S5P_TROPOMI_daily_{date_str.replace('-', '_')}.nc")
        if args.skip_existing and os.path.exists(nc_path):
            print(f"Skipping existing file {nc_path}")
            continue
        tif_path = export_day(date_str, args.out_dir, args.scale, india)
        convert_tif_to_netcdf(tif_path, nc_path)
        os.remove(tif_path)

    print('Sentinel-5P download and conversion completed.')


if __name__ == '__main__':
    main()
