import os
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
from sklearn.cluster import DBSCAN
from scipy.stats import pearsonr
import xarray as xr
from datetime import datetime, timedelta

def calculate_getis_ord_gi_star(grid, kernel_size=5):
    """
    Computes Getis-Ord Gi* z-scores for a 2D spatial grid using fast 2D convolution.
    Gi* = (Sum(w_ij * x_j) - X_bar * W_i) / (S * sqrt((n * W_ss - W_i^2) / (n - 1)))
    """
    ny, nx = grid.shape
    n = ny * nx
    
    valid_mask = ~np.isnan(grid)
    if not np.any(valid_mask):
        return np.zeros_like(grid)
        
    x_bar = np.nanmean(grid)
    s = np.nanstd(grid)
    if s == 0:
        return np.zeros_like(grid)
        
    # Fill NaNs with global mean for convolution calculation, then mask back later
    grid_filled = np.where(valid_mask, grid, x_bar)
    
    # Uniform weight kernel
    kernel = np.ones((kernel_size, kernel_size))
    
    # Local sums (sum_j w_ij * x_j)
    local_sum = ndimage.convolve(grid_filled, kernel, mode='constant', cval=x_bar)
    
    # Local weights sum (W_i). Reflects boundaries where cells are out of bounds
    local_w = ndimage.convolve(np.ones_like(grid_filled), kernel, mode='constant', cval=0.0)
    
    # Since weights are binary (1 or 0), sum of squared weights W_ss equals local_w
    local_w_ss = local_w
    
    # Numerator
    numerator = local_sum - (x_bar * local_w)
    
    # Denominator
    denom_inner = (n * local_w_ss - (local_w ** 2)) / (n - 1)
    denom_inner = np.maximum(denom_inner, 0.0)
    denominator = s * np.sqrt(denom_inner)
    
    z_score = np.zeros_like(grid)
    valid_denom = denominator > 0
    z_score[valid_denom] = numerator[valid_denom] / denominator[valid_denom]
    
    z_score[~valid_mask] = np.nan
    return z_score

def detect_dbscan_clusters(grid, lats, lons, threshold_std=1.5, eps_km=150.0, min_samples=3):
    """
    Identifies contiguous shapes of HCHO anomalies by running DBSCAN
    clustering on coordinates where HCHO exceeds (mean + threshold_std * std).
    """
    valid_mask = ~np.isnan(grid)
    if not np.any(valid_mask):
        return np.zeros_like(grid)
        
    mean_val = np.nanmean(grid)
    std_val = np.nanstd(grid)
    threshold = mean_val + (threshold_std * std_val)
    
    y_idx, x_idx = np.where((grid >= threshold) & valid_mask)
    if len(y_idx) == 0:
        return np.zeros_like(grid) - 2 # all below threshold
        
    points_lat = lats[y_idx]
    points_lon = lons[x_idx]
    coords = np.column_stack((points_lat, points_lon))
    
    # Haversine distance requires coordinates in radians
    coords_rad = np.radians(coords)
    eps_rad = eps_km / 6371.0 # earth radius is ~6371 km
    
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
    labels = db.fit_predict(coords_rad)
    
    # Create labels grid
    # -2 = below threshold, -1 = noise, >=0 = cluster ID
    label_grid = np.zeros_like(grid) - 2
    label_grid[y_idx, x_idx] = labels
    
    return label_grid

def analyze_fire_hcho_lag_correlation(base_dir=None):
    """
    Performs lag correlation analysis between daily fire counts (MODIS/VIIRS) 
    and spatial mean HCHO column densities in India (or specifically Northern India).
    """
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Performing Fire-HCHO Correlation and Lag-Correlation Analysis...")
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    # Load fires
    fire_df = pd.read_csv(os.path.join(raw_dir, "fire_alerts.csv"))
    fire_df['acq_date'] = pd.to_datetime(fire_df['acq_date'])
    
    # Daily fire count series
    daily_fires = fire_df.groupby('acq_date').size().reset_index(name='fire_count')
    daily_fires = daily_fires.sort_values('acq_date').reset_index(drop=True)
    
    # Daily mean HCHO series (IGP region: lat 24 to 32, lon 73 to 88)
    dates = sorted(daily_fires['acq_date'].dt.strftime("%Y-%m-%d").unique())
    daily_hcho = []
    
    from src.preprocessing import load_daily_netcdf
    for date_str in dates:
        ds = load_daily_netcdf(date_str, raw_dir)
        if ds is not None:
            # Subset to Northern India (Indo-Gangetic Plain) where biomass burning occurs
            igp = ds["HCHO"].sel(lat=slice(24, 32), lon=slice(73, 88))
            mean_hcho = float(igp.mean().values)
            daily_hcho.append({"date": pd.to_datetime(date_str), "mean_HCHO": mean_hcho})
            
    hcho_df = pd.DataFrame(daily_hcho)
    
    # Merge datasets
    merged = pd.merge(daily_fires, hcho_df, left_on='acq_date', right_on='date')
    
    # Lag correlation calculations
    lags = [0, 1, 2, 3, 5]
    lag_results = []
    
    for lag in lags:
        if lag == 0:
            r_val, p_val = pearsonr(merged['fire_count'], merged['mean_HCHO'])
        else:
            # Shift fire counts forward (meaning fire count leads HCHO concentration)
            shifted_fires = merged['fire_count'].shift(lag)
            valid_mask = ~shifted_fires.isna()
            r_val, p_val = pearsonr(shifted_fires[valid_mask], merged['mean_HCHO'][valid_mask])
            
        lag_results.append({
            "Lag_Days": lag,
            "Correlation": round(r_val, 4),
            "P_Value": round(p_val, 6)
        })
        
    lag_df = pd.DataFrame(lag_results)
    lag_df.to_csv(os.path.join(processed_dir, "fire_hcho_lag_correlation.csv"), index=False)
    
    print("\n--- Fire vs. HCHO Lag Correlation Results ---")
    print(lag_df.to_string(index=False))
    
    return lag_df, merged

def process_daily_hotspots(date_str, raw_dir, processed_dir):
    """
    Computes HCHO hotspot indicators (threshold, DBSCAN labels, Gi* z-scores)
    for a given day, and returns them packed in an xarray Dataset.
    """
    from src.preprocessing import load_daily_netcdf
    ds = load_daily_netcdf(date_str, raw_dir)
    if ds is None:
        return None
        
    hcho = ds["HCHO"].values
    lats = ds.lat.values
    lons = ds.lon.values
    
    # 1. Statistical thresholding (> mean + 2*std)
    mean_val = np.nanmean(hcho)
    std_val = np.nanstd(hcho)
    thresh = mean_val + 2.0 * std_val
    thresh_mask = np.where((hcho >= thresh) & (~np.isnan(hcho)), 1.0, 0.0)
    
    # 2. Getis-Ord Gi* z-scores
    gi_z = calculate_getis_ord_gi_star(hcho, kernel_size=5)
    
    # 3. DBSCAN clustering
    dbscan_labels = detect_dbscan_clusters(hcho, lats, lons, threshold_std=1.5, eps_km=150.0, min_samples=3)
    
    # Create Dataset
    out_ds = xr.Dataset(
        data_vars={
            "HCHO": (["lat", "lon"], hcho),
            "Threshold_Hotspots": (["lat", "lon"], thresh_mask),
            "Getis_Ord_Z": (["lat", "lon"], gi_z),
            "DBSCAN_Clusters": (["lat", "lon"], dbscan_labels)
        },
        coords={
            "lat": lats,
            "lon": lons,
            "time": pd.to_datetime(date_str)
        }
    )
    
    return out_ds

if __name__ == "__main__":

    analyze_fire_hcho_lag_correlation()

    from datetime import datetime
    raw_dir = "data/raw"
    processed_dir = "data/processed"

    ds = process_daily_hotspots(
        "2025-11-15",
        raw_dir,
        processed_dir
    )

    if ds is not None:
        out_file = "data/processed/hcho_hotspots_2025_11_15.nc"
        ds.to_netcdf(out_file)
        print(f"Hotspot file saved: {out_file}")