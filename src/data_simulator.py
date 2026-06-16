import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

def generate_spatial_fields(lats, lons, num_days, start_date):
    """
    Generates synthetic but meteorologically and physically consistent spatial fields
    for INSAT-3D AOD, Sentinel-5P, and ERA5 variables over India.
    """
    ny, nx = len(lats), len(lons)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Grid coordinates
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 1. Simulate ERA5 variables
    # Temperature: warmer in South, colder in North, seasonal cooling
    t2m_base = 300.0 - (lat_grid - 20.0) * 0.5  # Kelvin
    t2m_series = []
    
    # Relative Humidity: wetter near coasts (South/East), drier in NW (Rajasthan)
    r2_base = 50.0 + 20.0 * np.sin(lat_grid * np.pi / 40.0) - 20.0 * (lon_grid < 75) * ((lat_grid > 20) & (lat_grid < 30))
    r2_series = []
    
    # Boundary Layer Height: drops in winter, higher in south
    blh_base = 1200.0 - (lat_grid - 10.0) * 10.0
    blh_series = []
    
    # Winds: North-westerly winter monsoon (U > 0 and V < 0 in IGP)
    u10_base = np.zeros_like(lat_grid) + 2.0  # West to East
    v10_base = np.zeros_like(lat_grid) - 2.5  # North to South (so winds are NW-ly)
    u10_series = []
    v10_series = []

    # 2. Simulate Fires & HCHO / AOD / Gas columns
    # Active stubble burning in Punjab/Haryana (lat 29-32.5, lon 74-77) from Oct 20 to Nov 25
    fire_intensity = np.zeros((num_days, ny, nx))
    for t_idx, d in enumerate(dates):
        # Peak fire season: day 20 (Oct 20) to day 55 (Nov 25)
        if 20 <= t_idx <= 55:
            # Distance from Punjab center (30.8 N, 75.5 E)
            dist_punjab = np.sqrt((lat_grid - 30.8)**2 + (lon_grid - 75.5)**2)
            # Bell curve for fire intensity
            fire_fac = np.sin((t_idx - 20) / 35.0 * np.pi) # Peak in middle
            fire_intensity[t_idx] = 100.0 * fire_fac * np.exp(-dist_punjab**2 / 0.8)
            # Add some minor random fires elsewhere
            fire_intensity[t_idx] += np.random.exponential(0.5, size=(ny, nx)) * (np.random.rand(ny, nx) > 0.98)
        else:
            # Minor random fires
            fire_intensity[t_idx] = np.random.exponential(0.3, size=(ny, nx)) * (np.random.rand(ny, nx) > 0.98)
            
    # Calculate AOD and Pollutants
    # AOD: correlated with PM, has background + local fires + transported plume
    # Plume transported downwind (South-East direction: +dx, -dy)
    aod_series = []
    hcho_series = []
    no2_series = []
    so2_series = []
    co_series = []
    o3_series = []

    for t_idx, d in enumerate(dates):
        # Seasonal trend: background increases in winter (Nov/Dec) due to low BLH
        day_ratio = t_idx / num_days
        blh_val = blh_base * (1.0 - 0.4 * np.sin(day_ratio * np.pi / 2.0)) + np.random.normal(0, 50, (ny, nx))
        blh_val = np.maximum(blh_val, 150.0) # physical limit
        blh_series.append(blh_val)
        
        t2m_val = t2m_base - t_idx * 0.12 + np.random.normal(0, 1.0, (ny, nx))
        t2m_series.append(t2m_val)
        
        r2_val = np.clip(r2_base + np.random.normal(0, 5, (ny, nx)), 10, 100)
        r2_series.append(r2_val)
        
        # Wind variations
        u10_val = u10_base + np.random.normal(0, 0.5, (ny, nx))
        v10_val = v10_base + np.random.normal(0, 0.5, (ny, nx))
        u10_series.append(u10_val)
        v10_series.append(v10_val)
        
        # Plume computation (Gaussian dispersion along wind vector)
        plume = np.zeros((ny, nx))
        # Simple wind transport simulation for plume
        # Wind is NW, transport is SE
        for y in range(ny):
            for x in range(nx):
                intensity = fire_intensity[t_idx, y, x]
                if intensity > 1.0:
                    # propagate plume downwind
                    # wind speed ~ 3-4 m/s, let's say it disperses over ~1-3 degrees
                    # center shifts SE
                    dy = -1.5 * (u10_val[y, x] / 3.0)
                    dx = 1.5 * (-v10_val[y, x] / 3.0)
                    dist_plume = np.sqrt((lat_grid - (lats[y] + dy))**2 + (lon_grid - (lons[x] + dx))**2)
                    plume += intensity * np.exp(-dist_plume**2 / 1.5)
        
        # Background fields
        aod_bg = 0.15 + 0.1 * np.exp(-(lat_grid - 25)**2 / 100.0) # IGP belt high background
        # Scale background by low BLH (inversion)
        blh_factor = 1000.0 / blh_val
        aod_bg = aod_bg * blh_factor * 0.6
        
        aod_val = aod_bg + 0.005 * fire_intensity[t_idx] + 0.008 * plume + np.random.normal(0, 0.02, (ny, nx))
        aod_val = np.maximum(aod_val, 0.05)
        aod_series.append(aod_val)
        
        # HCHO: high near fires (biomass burning emission) + secondary production downwind
        # HCHO photolysis depends on temperature/sunlight
        hcho_bg = (1.5e-4 + 0.5e-4 * (t2m_val - 290) / 10.0)
        hcho_val = hcho_bg + 2.0e-5 * fire_intensity[t_idx] + 3.0e-5 * plume + np.random.normal(0, 1e-6, (ny, nx))
        hcho_val = np.maximum(hcho_val, 1e-6)
        hcho_series.append(hcho_val)
        
        # NO2: high in cities (Delhi, Mumbai, Kolkata, Singrauli)
        no2_bg = 2.0e-5 * np.exp(-((lat_grid - 28.6)**2 + (lon_grid - 77.2)**2)/0.5) # Delhi
        no2_bg += 1.8e-5 * np.exp(-((lat_grid - 19.1)**2 + (lon_grid - 72.9)**2)/0.5) # Mumbai
        no2_bg += 1.5e-5 * np.exp(-((lat_grid - 22.6)**2 + (lon_grid - 88.4)**2)/0.5) # Kolkata
        no2_bg += 1.2e-5 * np.exp(-((lat_grid - 13.0)**2 + (lon_grid - 80.3)**2)/0.5) # Chennai
        no2_bg += 1.0e-5 * np.exp(-((lat_grid - 24.1)**2 + (lon_grid - 82.7)**2)/0.3) # Singrauli (industrial)
        # general background scaled by BLH
        no2_val = (no2_bg * blh_factor + 1e-6 * fire_intensity[t_idx] + 
                   np.random.normal(1e-6, 2e-7, (ny, nx)))
        no2_val = np.maximum(no2_val, 1e-7)
        no2_series.append(no2_val)
        
        # SO2: high in industrial hotspots (Singrauli 24.1 N, 82.7 E, and others)
        so2_bg = 2.0e-5 * np.exp(-((lat_grid - 24.1)**2 + (lon_grid - 82.7)**2)/0.2) # Singrauli power plants
        so2_bg += 1.2e-5 * np.exp(-((lat_grid - 11.6)**2 + (lon_grid - 79.4)**2)/0.2) # Neyveli (lignite mine/power plant)
        so2_val = so2_bg * blh_factor + 2e-7 * fire_intensity[t_idx] + np.random.normal(2e-7, 5e-8, (ny, nx))
        so2_val = np.maximum(so2_val, 1e-8)
        so2_series.append(so2_val)
        
        # CO: correlates with fire plume + urban pollution
        co_bg = 0.05 + 0.03 * blh_factor
        co_val = co_bg + 0.005 * fire_intensity[t_idx] + 0.008 * plume + np.random.normal(0, 0.005, (ny, nx))
        co_val = np.maximum(co_val, 0.01)
        co_series.append(co_val)
        
        # O3: background + photochemistry (precursors: NO2 + VOCs like HCHO + temp)
        o3_bg = 0.03 + 0.005 * (t2m_val - 290.0) / 10.0
        o3_val = o3_bg + 0.1 * (no2_val * 1e5) * (hcho_val * 1e5) - 0.01 * (no2_val * 1e5) + np.random.normal(0, 0.002, (ny, nx))
        o3_val = np.maximum(o3_val, 0.01)
        o3_series.append(o3_val)
        
    return (dates, fire_intensity, {
        't2m': np.array(t2m_series), 'r2': np.array(r2_series), 'blh': np.array(blh_series),
        'u10': np.array(u10_series), 'v10': np.array(v10_series), 'aod': np.array(aod_series),
        'hcho': np.array(hcho_series), 'no2': np.array(no2_series), 'so2': np.array(so2_series),
        'co': np.array(co_series), 'o3': np.array(o3_series)
    })

def simulate_cpcb_stations(dates, gridded_data, lats, lons):
    """
    Simulates CPCB ground station daily measurements.
    """
    stations = [
        {"name": "Delhi (RK Puram)", "id": "STN01", "lat": 28.56, "lon": 77.18, "urban_factor": 1.5, "dust_factor": 1.0},
        {"name": "Patna (IGSC)", "id": "STN02", "lat": 25.61, "lon": 85.12, "urban_factor": 1.3, "dust_factor": 1.1},
        {"name": "Mumbai (Bandra)", "id": "STN03", "lat": 19.05, "lon": 72.82, "urban_factor": 1.2, "dust_factor": 0.8},
        {"name": "Bengaluru (Kengeri)", "id": "STN04", "lat": 12.91, "lon": 77.48, "urban_factor": 0.9, "dust_factor": 0.7},
        {"name": "Chennai (Adayar)", "id": "STN05", "lat": 13.01, "lon": 80.25, "urban_factor": 0.8, "dust_factor": 0.7},
        {"name": "Kolkata (Victoria)", "id": "STN06", "lat": 22.54, "lon": 88.34, "urban_factor": 1.1, "dust_factor": 0.8},
        {"name": "Amritsar (Golden Temple)", "id": "STN07", "lat": 31.62, "lon": 74.88, "urban_factor": 1.0, "dust_factor": 1.2},
        {"name": "Hisar (HAU)", "id": "STN08", "lat": 29.15, "lon": 75.70, "urban_factor": 0.8, "dust_factor": 1.5},
        {"name": "Lucknow (Lalbagh)", "id": "STN09", "lat": 26.85, "lon": 80.93, "urban_factor": 1.3, "dust_factor": 1.1},
        {"name": "Hyderabad (Sanathnagar)", "id": "STN10", "lat": 17.45, "lon": 78.43, "urban_factor": 1.0, "dust_factor": 0.7},
        {"name": "Guwahati (IIT)", "id": "STN11", "lat": 26.18, "lon": 91.69, "urban_factor": 0.7, "dust_factor": 0.9},
        {"name": "Jaipur (Shastri Nagar)", "id": "STN12", "lat": 26.93, "lon": 75.80, "urban_factor": 0.9, "dust_factor": 2.0}
    ]
    
    rows = []
    
    # Helper to find nearest grid index
    def get_grid_val(var_array, lat, lon):
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        return var_array[:, lat_idx, lon_idx]
    
    # Extract timeseries at station locations
    stn_data = {}
    for stn in stations:
        stn_data[stn['id']] = {
            'aod': get_grid_val(gridded_data['aod'], stn['lat'], stn['lon']),
            'no2_col': get_grid_val(gridded_data['no2'], stn['lat'], stn['lon']),
            'so2_col': get_grid_val(gridded_data['so2'], stn['lat'], stn['lon']),
            'co_col': get_grid_val(gridded_data['co'], stn['lat'], stn['lon']),
            'o3_col': get_grid_val(gridded_data['o3'], stn['lat'], stn['lon']),
            'blh': get_grid_val(gridded_data['blh'], stn['lat'], stn['lon']),
            't2m': get_grid_val(gridded_data['t2m'], stn['lat'], stn['lon']),
            'r2': get_grid_val(gridded_data['r2'], stn['lat'], stn['lon'])
        }

    for t_idx, date in enumerate(dates):
        for stn in stations:
            data = stn_data[stn['id']]
            aod = data['aod'][t_idx]
            no2_col = data['no2_col'][t_idx]
            so2_col = data['so2_col'][t_idx]
            co_col = data['co_col'][t_idx]
            o3_col = data['o3_col'][t_idx]
            blh = data['blh'][t_idx]
            t2m = data['t2m'][t_idx]
            r2 = data['r2'][t_idx]
            
            # Formulate surface concentrations from column densities and AOD
            # Surface PM2.5 (linked strongly to AOD, inversely to BLH)
            pm25 = aod * 150.0 * stn['urban_factor'] + (1000.0 / blh) * 10.0 + np.random.normal(0, 10)
            pm25 = np.maximum(pm25, 5.0)
            
            # PM10 (linked to PM2.5 + wind/dust factor)
            pm10 = pm25 * 1.6 + stn['dust_factor'] * 30.0 + np.random.normal(0, 15)
            pm10 = np.maximum(pm10, pm25 + 5.0)
            
            # Surface NO2 (correlated with column NO2)
            no2 = no2_col * 1e6 * 2.0 * stn['urban_factor'] + np.random.normal(0, 5)
            no2 = np.maximum(no2, 2.0)
            
            # Surface SO2
            so2 = so2_col * 1e6 * 1.5 * stn['urban_factor'] + np.random.normal(0, 2)
            so2 = np.maximum(so2, 1.0)
            
            # Surface CO (mg/m3)
            co = co_col * 15.0 * stn['urban_factor'] + np.random.normal(0, 0.2)
            co = np.maximum(co, 0.1)
            
            # Surface O3
            o3 = o3_col * 1.2e3 + (t2m - 290.0) * 1.5 - (r2 - 50.0) * 0.2 + np.random.normal(0, 5)
            o3 = np.maximum(o3, 5.0)
            
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "station_id": stn["id"],
                "station_name": stn["name"],
                "latitude": stn["lat"],
                "longitude": stn["lon"],
                "PM2.5": round(pm25, 2),
                "PM10": round(pm10, 2),
                "NO2": round(no2, 2),
                "SO2": round(so2, 2),
                "CO": round(co, 2),
                "O3": round(o3, 2)
            })
            
    df = pd.DataFrame(rows)
    return df

def generate_modis_viirs_fires(dates, fire_intensity, lats, lons):
    """
    Generates simulated MODIS/VIIRS active fire alerts.
    """
    rows = []
    ny, nx = len(lats), len(lons)
    
    for t_idx, date in enumerate(dates):
        # Find cells with fire activity
        y_indices, x_indices = np.where(fire_intensity[t_idx] > 2.0)
        
        for y, x in zip(y_indices, x_indices):
            intensity = fire_intensity[t_idx, y, x]
            # Multiple fire pixels per active grid cell depending on intensity
            num_pixels = int(np.clip(intensity / 8.0, 1, 15))
            
            for _ in range(num_pixels):
                # Add small random spatial jitter within the 0.5x0.5 degree grid cell
                lat_jitter = lats[y] + np.random.uniform(-0.23, 0.23)
                lon_jitter = lons[x] + np.random.uniform(-0.23, 0.23)
                
                frp = intensity * np.random.uniform(5.0, 20.0)
                satellite = "MODIS" if np.random.rand() > 0.5 else "VIIRS"
                confidence = np.random.choice([50, 70, 90, 100], p=[0.1, 0.3, 0.4, 0.2])
                
                rows.append({
                    "latitude": round(lat_jitter, 4),
                    "longitude": round(lon_jitter, 4),
                    "acq_date": date.strftime("%Y-%m-%d"),
                    "frp": round(frp, 1),
                    "satellite": satellite,
                    "confidence": int(confidence)
                })
                
    df = pd.DataFrame(rows)
    return df

def save_gridded_netcdf(dates, variables, lats, lons, raw_dir):
    """
    Saves the simulated daily gridded datasets to NetCDF files grouped by source.
    """
    # Create raw folders if they don't exist
    insat_dir = os.path.join(raw_dir, "insat3d")
    s5p_dir = os.path.join(raw_dir, "s5p")
    era5_dir = os.path.join(raw_dir, "era5")
    
    for d in [insat_dir, s5p_dir, era5_dir]:
        os.makedirs(d, exist_ok=True)
        
    print("Writing netcdf files...")
    
    # Iterate over each day and write NC files
    for t_idx, date in enumerate(dates):
        date_str = date.strftime("%Y_%m_%d")
        
        # 1. INSAT-3D AOD
        ds_insat = xr.Dataset(
            data_vars={
                "AOD": (["lat", "lon"], variables["aod"][t_idx])
            },
            coords={
                "lat": lats,
                "lon": lons,
                "time": pd.to_datetime(date)
            }
        )
        ds_insat.to_netcdf(os.path.join(insat_dir, f"INSAT3D_AOD_daily_{date_str}.nc"))
        
        # 2. Sentinel-5P TROPOMI
        ds_s5p = xr.Dataset(
            data_vars={
                "NO2": (["lat", "lon"], variables["no2"][t_idx]),
                "SO2": (["lat", "lon"], variables["so2"][t_idx]),
                "CO": (["lat", "lon"], variables["co"][t_idx]),
                "O3": (["lat", "lon"], variables["o3"][t_idx]),
                "HCHO": (["lat", "lon"], variables["hcho"][t_idx]),
            },
            coords={
                "lat": lats,
                "lon": lons,
                "time": pd.to_datetime(date)
            }
        )
        ds_s5p.to_netcdf(os.path.join(s5p_dir, f"S5P_TROPOMI_daily_{date_str}.nc"))
        
        # 3. ERA5 Meteorology
        ds_era5 = xr.Dataset(
            data_vars={
                "t2m": (["lat", "lon"], variables["t2m"][t_idx]),
                "r2": (["lat", "lon"], variables["r2"][t_idx]),
                "blh": (["lat", "lon"], variables["blh"][t_idx]),
                "u10": (["lat", "lon"], variables["u10"][t_idx]),
                "v10": (["lat", "lon"], variables["v10"][t_idx]),
            },
            coords={
                "lat": lats,
                "lon": lons,
                "time": pd.to_datetime(date)
            }
        )
        ds_era5.to_netcdf(os.path.join(era5_dir, f"ERA5_daily_{date_str}.nc"))

def run_simulation(base_dir="c:/Users/Anjali/OneDrive/Desktop/ISRO"):
    print("Starting Spatial-Temporal Data Simulation...")
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # 0.5 degree grid over India
    lats = np.arange(8.0, 38.0, 0.5)
    lons = np.arange(68.0, 98.0, 0.5)
    
    start_date = datetime(2025, 10, 1)
    num_days = 90  # Oct 1 to Dec 29, 2025
    
    dates, fire_intensity, gridded_data = generate_spatial_fields(lats, lons, num_days, start_date)
    
    # Generate CPCB ground station measurements
    print("Simulating CPCB Ground Stations...")
    cpcb_df = simulate_cpcb_stations(dates, gridded_data, lats, lons)
    cpcb_df.to_csv(os.path.join(raw_dir, "cpcb_stations.csv"), index=False)
    
    # Generate active fire counts
    print("Simulating MODIS/VIIRS Active Fires...")
    fire_df = generate_modis_viirs_fires(dates, fire_intensity, lats, lons)
    fire_df.to_csv(os.path.join(raw_dir, "fire_alerts.csv"), index=False)
    
    # Save gridded datasets to NetCDF
    print("Saving gridded datasets to NetCDF...")
    save_gridded_netcdf(dates, gridded_data, lats, lons, raw_dir)
    print("Simulation complete! Raw data successfully created.")

if __name__ == "__main__":
    run_simulation()
