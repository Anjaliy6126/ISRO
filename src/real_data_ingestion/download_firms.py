import os
import argparse
import requests
import pandas as pd

def download_firms_fires(api_key, date_str, output_path):
    """
    Downloads active fire counts for India from NASA's FIRMS API.
    BBox: [68, 8, 98, 38] (West, South, East, North)
    """
    # India bounding box: [min_lon, min_lat, max_lon, max_lat]
    bbox_str = "68,8,98,38"
    
    # Sources: MODIS_SP (Standard Processing), VIIRS_SNPP_SP, VIIRS_NOAA20_SP
    source = "MODIS_SP"
    
    # 1-day range query
    range_days = 1
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{bbox_str}/{range_days}/{date_str}"
    
    print(f"Requesting active fires from NASA FIRMS API for {date_str}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        # Check if directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save raw CSV
        with open(output_path, 'w') as f:
            f.write(response.text)
            
        print(f"Active fires saved successfully to {output_path}")
        
        # Display some info
        try:
            df = pd.read_csv(output_path)
            print(f"Found {len(df)} fire pixels on this day.")
            if len(df) > 0:
                print(df[['latitude', 'longitude', 'frp', 'confidence']].head())
        except Exception as e:
            pass
    else:
        print(f"Failed to download active fires: HTTP {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download active fire alerts from NASA FIRMS API.")
    parser.add_argument("--api_key", type=str, required=True, help="Your NASA FIRMS MAPS API Key.")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD format.")
    parser.add_argument("--out_file", type=str, default="c:/Users/Anjali/OneDrive/Desktop/ISRO/data/raw/fire_alerts.csv", help="Target output filepath.")
    args = parser.parse_args()
    
    download_firms_fires(args.api_key, args.date, args.out_file)
