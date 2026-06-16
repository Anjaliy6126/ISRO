import os
import argparse
from datetime import datetime, timedelta
import ee
import geemap

def download_s5p(date_str, output_dir):
    """
    Connects to Google Earth Engine, averages Sentinel-5P parameters,
    clips them to India, and exports a unified GeoTIFF/NetCDF grid.
    """
    # Initialize GEE
    try:
        ee.Initialize()
    except Exception as e:
        print("GEE not initialized. Attempting authentication...")
        ee.Authenticate()
        ee.Initialize()
        
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    date_formatted = dt.strftime("%Y_%m_%d")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"S5P_TROPOMI_daily_{date_formatted}.tif")
    
    # Boundary collection for India
    india = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('adm0_name', 'India'))
    
    start_date = ee.Date(date_str)
    end_date = start_date.advance(1, 'day')
    
    # Define parameters, GEE collection ids, and band names
    params = {
        'NO2': ('COPERNICUS/S5P/OFFL/L3_NO2', 'NO2_column_number_density'),
        'SO2': ('COPERNICUS/S5P/OFFL/L3_SO2', 'SO2_column_number_density'),
        'CO': ('COPERNICUS/S5P/OFFL/L3_CO', 'CO_column_number_density'),
        'O3': ('COPERNICUS/S5P/OFFL/L3_O3', 'O3_column_number_density'),
        'HCHO': ('COPERNICUS/S5P/OFFL/L3_HCHO', 'tropospheric_HCHO_column_number_density')
    }
    
    images = []
    for param_name, (coll_id, band_name) in params.items():
        img = ee.ImageCollection(coll_id) \
                .select(band_name) \
                .filterDate(start_date, end_date) \
                .mean() \
                .clip(india) \
                .rename(param_name)
        images.append(img)
        
    # Combine individual parameter images into a single multi-band image
    combined_image = ee.Image.cat(images)
    
    print(f"Exporting Sentinel-5P gridded image for {date_str} to {output_path}...")
    
    # Export to local file using geemap (scale: 10km grid resolution)
    geemap.ee_export_image(
        combined_image,
        filename=output_path,
        scale=10000, 
        region=india.geometry(),
        file_per_band=False
    )
    print("S5P TROPOMI Ingestion Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Sentinel-5P gridded variables from Google Earth Engine.")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD format.")
    parser.add_argument("--out_dir", type=str, default="c:/Users/Anjali/OneDrive/Desktop/ISRO/data/raw/s5p", help="Target output directory.")
    args = parser.parse_args()
    
    download_s5p(args.date, args.out_dir)
