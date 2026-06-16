import os
import argparse
from datetime import datetime
import cdsapi

def download_era5(date_str, output_dir):
    """
    Downloads ERA5 single levels meteorological reanalysis data for India.
    BBox: [38, 68, 8, 98] (North, West, South, East)
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    date_formatted = dt.strftime("%Y_%m_%d")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ERA5_daily_{date_formatted}.nc")
    
    print(f"Requesting ERA5 data for {date_str}...")
    
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature', 
                '2m_dewpoint_temperature', # can be used to calculate RH
                'boundary_layer_height',
                '10m_u_component_of_wind', 
                '10m_v_component_of_wind'
            ],
            'year': str(dt.year),
            'month': f"{dt.month:02d}",
            'day': f"{dt.day:02d}",
            'time': '12:00', # midday snapshot, can request full list of hours
            'area': [38.0, 68.0, 8.0, 98.0], # N, W, S, E bounds
        },
        output_path
    )
    print(f"Saved ERA5 netcdf to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 NetCDF grids for India.")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD format.")
    parser.add_argument("--out_dir", type=str, default="c:/Users/Anjali/OneDrive/Desktop/ISRO/data/raw/era5", help="Target output directory.")
    args = parser.parse_args()
    
    try:
        download_era5(args.date, args.out_dir)
    except Exception as e:
        print(f"Error calling CDS API: {e}")
        print("Verify your ~/.cdsapirc file has correct credentials.")
