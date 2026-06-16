import os
import argparse
import numpy as np
import xarray as xr
import h5py

def extract_mosdac_hdf5_to_nc(h5_file_path, output_nc_path):
    """
    Reads a raw MOSDAC INSAT-3D Level-3 HDF5 file, extracts lat, lon, and AOD grid,
    and saves it to NetCDF matching the project standards.
    """
    print(f"Reading HDF5 INSAT-3D file from {h5_file_path}...")
    
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"HDF5 file not found at {h5_file_path}")
        
    with h5py.File(h5_file_path, 'r') as f:
        # Inspecting structure. Typically MOSDAC INSAT-3D Level-3 AOD datasets
        # store variables directly under root or inside groups.
        # Let's read standard dataset names:
        # Latitude: 'Latitude' or 'lat'
        # Longitude: 'Longitude' or 'lon'
        # AOD: 'Aerosol_Optical_Depth' or 'AOD'
        
        # Try different possible keys used in MOSDAC formats
        lat_key = 'Latitude' if 'Latitude' in f.keys() else 'lat'
        lon_key = 'Longitude' if 'Longitude' in f.keys() else 'lon'
        aod_key = 'AOD' if 'AOD' in f.keys() else 'Aerosol_Optical_Depth'
        
        if lat_key not in f or lon_key not in f or aod_key not in f:
            print("Warning: Standard keys not found. Available keys in HDF5:")
            print(list(f.keys()))
            raise KeyError("Specified datasets (Latitude, Longitude, AOD) not found in file.")
            
        lat = f[lat_key][:]
        lon = f[lon_key][:]
        aod = f[aod_key][:]
        
        # Quality control filter: replace missing values/fill values (e.g. -999.0) with NaN
        fill_val = f[aod_key].attrs.get('_FillValue', -999.0)
        aod = np.where(aod == fill_val, np.nan, aod)
        
        # If AOD is scaled in HDF5 (e.g., stored as integers), apply scale factor and offset
        scale = f[aod_key].attrs.get('scale_factor', 1.0)
        offset = f[aod_key].attrs.get('add_offset', 0.0)
        aod = aod * scale + offset
        
        # Build standard xarray Dataset matching project coordinates
        # Bounding box is adjusted to align with our standard grid
        ds = xr.Dataset(
            data_vars={
                "AOD": (["lat", "lon"], aod)
            },
            coords={
                "lat": lat,
                "lon": lon
            }
        )
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_nc_path), exist_ok=True)
        ds.to_netcdf(output_nc_path)
        print(f"Successfully extracted AOD and saved NetCDF to {output_nc_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and convert MOSDAC INSAT-3D HDF5 AOD files to NetCDF.")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to raw MOSDAC HDF5 file.")
    parser.add_argument("--out_path", type=str, required=True, help="Target path to save NetCDF (.nc) file.")
    args = parser.parse_args()
    
    try:
        extract_mosdac_hdf5_to_nc(args.h5_path, args.out_path)
    except Exception as e:
        print(f"Error extracting HDF5: {e}")
