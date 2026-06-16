import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime

def load_daily_netcdf(date_str, raw_dir):
    """
    Loads raw INSAT-3D, Sentinel-5P, and ERA5 NetCDF files for a specific date,
    and merges them into a single gridded xarray Dataset.
    """
    date_formatted = date_str.replace("-", "_")
    insat_pattern = os.path.join(raw_dir, "insat3d", f"INSAT3D_AOD_daily_{date_formatted}.nc")
    s5p_pattern = os.path.join(raw_dir, "s5p", f"S5P_TROPOMI_daily_{date_formatted}.nc")
    era5_pattern = os.path.join(raw_dir, "era5", f"ERA5_daily_{date_formatted}.nc")
    
    # Check if files exist
    if not (os.path.exists(insat_pattern) and os.path.exists(s5p_pattern) and os.path.exists(era5_pattern)):
        return None
        
    # Open datasets
    ds_insat = xr.open_dataset(insat_pattern)
    ds_s5p = xr.open_dataset(s5p_pattern)
    ds_era5 = xr.open_dataset(era5_pattern)
    
    # Merge datasets. They are already on the same grid in simulation,
    # but in a real-world script we would interpolate to a common grid.
    # To handle general grids, we interpolate INSAT and ERA5 to Sentinel-5P's grid.
    ds_insat_interp = ds_insat.interp_like(ds_s5p, method="nearest")
    ds_era5_interp = ds_era5.interp_like(ds_s5p, method="nearest")
    
    merged = xr.merge([ds_insat_interp, ds_s5p, ds_era5_interp])
    # Drop time coord from variables to avoid dimensional conflict in alignment, keeping it as attribute or coordinate
    return merged

def align_and_match(base_dir="c:/Users/Anjali/OneDrive/Desktop/ISRO"):
    """
    Executes spatiotemporal matching: matches ground CPCB station observations
    with satellite columns and meteorological grid cells on a daily basis.
    """
    print("Starting Spatiotemporal Matching...")
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    cpcb_path = os.path.join(raw_dir, "cpcb_stations.csv")
    if not os.path.exists(cpcb_path):
        raise FileNotFoundError(f"CPCB stations file not found at {cpcb_path}. Run simulation first.")
        
    cpcb_df = pd.read_csv(cpcb_path)
    cpcb_df['date'] = pd.to_datetime(cpcb_df['date'])
    
    unique_dates = cpcb_df['date'].dt.strftime("%Y-%m-%d").unique()
    matched_rows = []
    
    # Store daily merged datasets in a dict to avoid reloading
    daily_grids = {}
    
    for date_str in unique_dates:
        ds_merged = load_daily_netcdf(date_str, raw_dir)
        if ds_merged is not None:
            daily_grids[date_str] = ds_merged
            
    print(f"Loaded {len(daily_grids)} days of satellite and meteorological grids.")
    
    # Match stations
    for idx, row in cpcb_df.iterrows():
        date_str = row['date'].strftime("%Y-%m-%d")
        if date_str not in daily_grids:
            continue
            
        ds = daily_grids[date_str]
        
        # Nearest neighbor lookup in xarray
        stn_lat = row['latitude']
        stn_lon = row['longitude']
        
        grid_point = ds.sel(lat=stn_lat, lon=stn_lon, method="nearest")
        
        # Extract features
        matched_data = row.to_dict()
        matched_data.update({
            "AOD": float(grid_point["AOD"].values),
            "col_NO2": float(grid_point["NO2"].values),
            "col_SO2": float(grid_point["SO2"].values),
            "col_CO": float(grid_point["CO"].values),
            "col_O3": float(grid_point["O3"].values),
            "col_HCHO": float(grid_point["HCHO"].values),
            "t2m": float(grid_point["t2m"].values),
            "r2": float(grid_point["r2"].values),
            "blh": float(grid_point["blh"].values),
            "u10": float(grid_point["u10"].values),
            "v10": float(grid_point["v10"].values)
        })
        
        matched_rows.append(matched_data)
        
    matched_df = pd.DataFrame(matched_rows)
    matched_df['date'] = pd.to_datetime(matched_df['date'])
    
    # Generate Lagged Features per station
    print("Creating Lagged and Temporal Features...")
    matched_df = matched_df.sort_values(by=["station_id", "date"]).reset_index(drop=True)
    
    # 1-day lag features for satellite proxy
    matched_df["AOD_lag1"] = matched_df.groupby("station_id")["AOD"].shift(1)
    matched_df["col_HCHO_lag1"] = matched_df.groupby("station_id")["col_HCHO"].shift(1)
    
    # Temporal features
    matched_df["day_of_year"] = matched_df["date"].dt.dayofyear
    matched_df["month"] = matched_df["date"].dt.month
    
    # Drop rows with NaN in lags (first day of each station)
    matched_df = matched_df.dropna().reset_index(drop=True)
    
    # Save matched dataset
    processed_path = os.path.join(processed_dir, "matched_station_data.csv")
    matched_df.to_csv(processed_path, index=False)
    print(f"Spatiotemporal matched dataset saved to {processed_path}. Shape: {matched_df.shape}")
    
    return matched_df

if __name__ == "__main__":
    align_and_match()
