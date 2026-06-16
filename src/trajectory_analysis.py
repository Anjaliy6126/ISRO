import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_back_trajectory(start_lat, start_lon, start_date_str, raw_dir, duration_hours=48, dt_hours=1):
    """
    Computes a kinematic back-trajectory from a starting coordinates (receptor) and date.
    Traces hourly steps backward by reading wind components U, V.
    """
    R = 6371000.0  # Earth radius in meters
    current_lat = start_lat
    current_lon = start_lon
    current_time = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    path = [{
        "hour": 0,
        "datetime": current_time,
        "latitude": current_lat,
        "longitude": current_lon,
        "u10": 0.0,
        "v10": 0.0
    }]
    
    # Store loaded datasets in a local cache to avoid redundant disk I/O
    grid_cache = {}
    
    from src.preprocessing import load_daily_netcdf
    
    for step in range(1, duration_hours + 1):
        # Determine which date's wind grid to load
        date_str = current_time.strftime("%Y-%m-%d")
        
        if date_str not in grid_cache:
            ds = load_daily_netcdf(date_str, raw_dir)
            if ds is not None:
                grid_cache[date_str] = ds
            else:
                # If we go out of bounds of our available data dates, stop tracing
                break
                
        ds = grid_cache[date_str]
        
        # Check boundary coordinates
        min_lat, max_lat = float(ds.lat.min()), float(ds.lat.max())
        min_lon, max_lon = float(ds.lon.min()), float(ds.lon.max())
        
        if not (min_lat <= current_lat <= max_lat and min_lon <= current_lon <= max_lon):
            break # Trajectory left India bounding box
            
        # Sel nearest wind values
        grid_pt = ds.sel(lat=current_lat, lon=current_lon, method="nearest")
        u = float(grid_pt["u10"].values)
        v = float(grid_pt["v10"].values)
        
        # Calculate distance traveled backward (negative sign for backward tracking)
        # dt is in hours, convert to seconds
        dt_seconds = dt_hours * 3600.0
        dy = -v * dt_seconds
        dx = -u * dt_seconds
        
        # Convert meter changes to degree changes
        dlat = (dy / R) * (180.0 / np.pi)
        # Prevent division by zero if close to pole
        cos_lat = np.cos(np.radians(current_lat))
        if cos_lat < 0.01:
            cos_lat = 0.01
        dlon = (dx / (R * cos_lat)) * (180.0 / np.pi)
        
        # Update positions
        current_lat += dlat
        current_lon += dlon
        current_time -= timedelta(hours=dt_hours)
        
        path.append({
            "hour": -step * dt_hours,
            "datetime": current_time,
            "latitude": round(current_lat, 4),
            "longitude": round(current_lon, 4),
            "u10": round(u, 2),
            "v10": round(v, 2)
        })
        
    df = pd.DataFrame(path)
    return df

def generate_wind_field_plot(date_str, raw_dir, output_path=None):
    """
    Plots the wind vector quiver map over India for a specific date.
    """
    from src.preprocessing import load_daily_netcdf
    ds = load_daily_netcdf(date_str, raw_dir)
    if ds is None:
        return None
        
    # Squeeze or select time
    u = ds["u10"].values
    v = ds["v10"].values
    lats = ds.lat.values
    lons = ds.lon.values
    
    # Subsample grid for clearer vector arrow visualization
    skip = 3
    lons_sub, lats_sub = np.meshgrid(lons[::skip], lats[::skip])
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Plot wind speed background
    wind_speed = np.sqrt(u**2 + v**2)
    c = ax.contourf(lons, lats, wind_speed, cmap="Blues", alpha=0.6, levels=15)
    cbar = fig.colorbar(c, ax=ax, label="Wind Speed (m/s)")
    
    # Quiver plot
    ax.quiver(lons_sub, lats_sub, u_sub, v_sub, color="navy", scale=100, width=0.002, alpha=0.8)
    
    ax.set_title(f"10m Wind Fields over India - {date_str}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Boundaries
    ax.set_xlim(68, 98)
    ax.set_ylim(8, 38)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        return fig
        
def generate_trajectory_plot(trajectory_df, fire_df, date_str, output_path=None):
    """
    Generates a map showing back-trajectory path and active fire counts.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Plot active fires on that day
    # Filter fire alerts to a window of the past 2 days to show accumulated burning
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    date_limit = dt - timedelta(days=2)
    fire_subset = fire_df[(pd.to_datetime(fire_df['acq_date']) >= date_limit) & 
                          (pd.to_datetime(fire_df['acq_date']) <= dt)]
                          
    if len(fire_subset) > 0:
        ax.scatter(fire_subset['longitude'], fire_subset['latitude'], 
                   c='orange', s=fire_subset['frp']/10.0, alpha=0.4, label='MODIS/VIIRS Active Fires', edgecolors='red', linewidths=0.2)
                   
    # Plot back trajectory path
    ax.plot(trajectory_df['longitude'], trajectory_df['latitude'], 
            color='darkred', linewidth=2.5, marker='o', markersize=4, label='Air Mass Back-Trajectory (48h)')
            
    # Highlight start receptor point (hour = 0)
    start_pt = trajectory_df.iloc[0]
    ax.plot(start_pt['longitude'], start_pt['latitude'], 
            color='black', marker='*', markersize=12, label='Receptor City')
            
    # Highlight end point of trajectory
    end_pt = trajectory_df.iloc[-1]
    ax.plot(end_pt['longitude'], end_pt['latitude'], 
            color='blue', marker='X', markersize=8, label='Source Air Mass (48h ago)')
            
    # Text annotation hourly progress
    for idx, row in trajectory_df.iterrows():
        if row['hour'] in [0, -12, -24, -36, -48]:
            ax.annotate(f"{int(row['hour'])}h", (row['longitude'], row['latitude']),
                        textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, weight='bold', color='darkred')
                        
    ax.set_title(f"Air Transport Pathway & Fire Detections - Receptor Date: {date_str}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(68, 98)
    ax.set_ylim(8, 38)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        return fig
