import os
import pandas as pd
import xarray as xr


def load_daily_netcdf(date_str, raw_dir):
    """
    Loads raw INSAT-3D, Sentinel-5P, and ERA5 NetCDF files for a specific date,
    and merges them into a single gridded xarray Dataset.
    """

    date_formatted = date_str.replace("-", "_")

    insat_file = os.path.join(
        raw_dir,
        "insat3d",
        f"INSAT3D_AOD_daily_{date_formatted}.nc"
    )

    s5p_file = os.path.join(
        raw_dir,
        "s5p",
        f"S5P_TROPOMI_daily_{date_formatted}.nc"
    )

    era5_file = os.path.join(
        raw_dir,
        "era5",
        f"ERA5_daily_{date_formatted}.nc"
    )

    # Check files exist
    if not (
        os.path.exists(insat_file)
        and os.path.exists(s5p_file)
        and os.path.exists(era5_file)
    ):
        return None

    # Open datasets
    ds_insat = xr.open_dataset(insat_file)
    ds_s5p = xr.open_dataset(s5p_file)
    ds_era5 = xr.open_dataset(era5_file)

    # Interpolate INSAT and ERA5 to Sentinel-5P grid
    ds_insat_interp = ds_insat.interp_like(
        ds_s5p,
        method="nearest"
    )

    ds_era5_interp = ds_era5.interp_like(
        ds_s5p,
        method="nearest"
    )

    # Merge datasets
    merged = xr.merge(
        [ds_insat_interp, ds_s5p, ds_era5_interp],
        compat="override"
    )

    return merged


def align_and_match(base_dir=None):
    """
    Executes spatiotemporal matching:
    Matches CPCB stations with satellite and ERA5 grid cells.
    """

    if base_dir is None:
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

    print("Starting Spatiotemporal Matching...")

    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")

    os.makedirs(processed_dir, exist_ok=True)

    cpcb_path = os.path.join(raw_dir, "cpcb_stations.csv")

    if not os.path.exists(cpcb_path):
        raise FileNotFoundError(
            f"CPCB stations file not found at {cpcb_path}"
        )

    cpcb_df = pd.read_csv(cpcb_path)
    cpcb_df["date"] = pd.to_datetime(cpcb_df["date"])

    unique_dates = (
        cpcb_df["date"]
        .dt.strftime("%Y-%m-%d")
        .unique()
    )

    matched_rows = []
    daily_grids = {}

    # Load all daily merged datasets
    for date_str in unique_dates:

        ds = load_daily_netcdf(date_str, raw_dir)

        if ds is not None:
            daily_grids[date_str] = ds

    print(f"Loaded {len(daily_grids)} days of satellite and meteorological grids.")

    # Match each station
    for _, row in cpcb_df.iterrows():

        date_str = row["date"].strftime("%Y-%m-%d")

        if date_str not in daily_grids:
            continue

        ds = daily_grids[date_str]

        grid_point = ds.sel(
            lat=row["latitude"],
            lon=row["longitude"],
            method="nearest"
        )

        matched = row.to_dict()

        matched.update({

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

        matched_rows.append(matched)

    matched_df = pd.DataFrame(matched_rows)

    matched_df["date"] = pd.to_datetime(
        matched_df["date"]
    )

    print("Creating Lagged and Temporal Features...")

    matched_df = (
        matched_df
        .sort_values(
            by=["station_id", "date"]
        )
        .reset_index(drop=True)
    )

    matched_df["AOD_lag1"] = (
        matched_df
        .groupby("station_id")["AOD"]
        .shift(1)
    )

    matched_df["col_HCHO_lag1"] = (
        matched_df
        .groupby("station_id")["col_HCHO"]
        .shift(1)
    )

    matched_df["day_of_year"] = (
        matched_df["date"]
        .dt.dayofyear
    )

    matched_df["month"] = (
        matched_df["date"]
        .dt.month
    )

    matched_df = (
        matched_df
        .dropna()
        .reset_index(drop=True)
    )

    output_file = os.path.join(
        processed_dir,
        "matched_station_data.csv"
    )

    matched_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Spatiotemporal matched dataset saved to {output_file}"
    )

    print(
        f"Dataset Shape: {matched_df.shape}"
    )

    return matched_df


if __name__ == "__main__":
    align_and_match()