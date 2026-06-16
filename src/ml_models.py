import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xarray as xr
from datetime import datetime, timedelta

# Features and targets defined based on data matching
FEATURE_COLS = [
    'AOD', 'col_NO2', 'col_SO2', 'col_CO', 'col_O3', 'col_HCHO', 
    't2m', 'r2', 'blh', 'u10', 'v10', 'AOD_lag1', 'col_HCHO_lag1',
    'day_of_year', 'month'
]

TARGET_COLS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

def train_and_evaluate_ml(base_dir="c:/Users/Anjali/OneDrive/Desktop/ISRO"):
    """
    Loads matched dataset, performs temporal train-test split,
    trains Random Forest and XGBoost for each pollutant, and calculates metrics.
    """
    print("Training and Evaluating Machine Learning Models...")
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    df_path = os.path.join(processed_dir, "matched_station_data.csv")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Processed matched dataset not found at {df_path}. Run preprocessing first.")
        
    df = pd.read_csv(df_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Temporal split: Train on first 70% of days, test on last 30% of days
    unique_dates = sorted(df['date'].unique())
    split_idx = int(len(unique_dates) * 0.7)
    split_date = unique_dates[split_idx]
    
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()
    
    print(f"Training on records before {split_date.strftime('%Y-%m-%d')} (N={len(train_df)})")
    print(f"Testing on records from {split_date.strftime('%Y-%m-%d')} onwards (N={len(test_df)})")
    
    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]
    
    metrics_summary = []
    
    for pollutant in TARGET_COLS:
        y_train = train_df[pollutant]
        y_test = test_df[pollutant]
        
        # 1. Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # 2. XGBoost Regressor
        xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        # Save models
        joblib.dump(rf_model, os.path.join(models_dir, f"rf_{pollutant}.joblib"))
        joblib.dump(xgb_model, os.path.join(models_dir, f"xgb_{pollutant}.joblib"))
        
        # Evaluate RF
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_r, _ = pearsonr(y_test, rf_pred)
        
        # Evaluate XGB
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_r, _ = pearsonr(y_test, xgb_pred)
        
        metrics_summary.append({
            "pollutant": pollutant,
            "model": "Random Forest",
            "RMSE": round(rf_rmse, 3),
            "MAE": round(rf_mae, 3),
            "R2": round(rf_r2, 3),
            "Pearson_R": round(rf_r, 3)
        })
        
        metrics_summary.append({
            "pollutant": pollutant,
            "model": "XGBoost",
            "RMSE": round(xgb_rmse, 3),
            "MAE": round(xgb_mae, 3),
            "R2": round(xgb_r2, 3),
            "Pearson_R": round(xgb_r, 3)
        })
        
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(os.path.join(processed_dir, "ml_metrics.csv"), index=False)
    
    print("\n--- ML Model Validation Metrics (Temporal Test Set) ---")
    print(metrics_df.to_string(index=False))
    
    return metrics_df

def predict_surface_grid(date_str, models_dir, raw_dir, model_type="xgb"):
    """
    Loads raw NetCDF grids for a day, predicts surface concentration for all 6 CPCB pollutants
    across the entire India grid, and returns an xarray Dataset.
    """
    from src.preprocessing import load_daily_netcdf
    from src.aqi_calculator import calculate_cpcb_aqi, get_aqi_category
    from datetime import datetime, timedelta
    
    ds = load_daily_netcdf(date_str, raw_dir)
    if ds is None:
        return None
        
    ny, nx = len(ds.lat), len(ds.lon)
    
    # Prepare features for the grid
    # Flattens spatial grid into 1D array of pixels for model prediction
    grid_lat, grid_lon = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
    
    # Date variables
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = dt.timetuple().tm_yday
    month = dt.month
    
    # We need lag values. In a grid, for simplicity, we use the current day AOD and HCHO as proxies
    # or look up yesterday's file. To make this self-contained, we will approximate AOD_lag1 and col_HCHO_lag1
    # with a slightly perturbed version (or if yesterday's file exists, we load it).
    yesterday = dt - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    ds_yest = load_daily_netcdf(yesterday_str, raw_dir)
    
    if ds_yest is not None:
        aod_lag1 = ds_yest["AOD"].values
        hcho_lag1 = ds_yest["HCHO"].values
    else:
        # Fallback to current day if first day of simulation
        aod_lag1 = ds["AOD"].values * 0.95
        hcho_lag1 = ds["HCHO"].values * 0.95
        
    # Build dataframe for grid predictions
    grid_data = {
        'AOD': ds["AOD"].values.flatten(),
        'col_NO2': ds["NO2"].values.flatten(),
        'col_SO2': ds["SO2"].values.flatten(),
        'col_CO': ds["CO"].values.flatten(),
        'col_O3': ds["O3"].values.flatten(),
        'col_HCHO': ds["HCHO"].values.flatten(),
        't2m': ds["t2m"].values.flatten(),
        'r2': ds["r2"].values.flatten(),
        'blh': ds["blh"].values.flatten(),
        'u10': ds["u10"].values.flatten(),
        'v10': ds["v10"].values.flatten(),
        'AOD_lag1': aod_lag1.flatten(),
        'col_HCHO_lag1': hcho_lag1.flatten(),
        'day_of_year': np.full(ny * nx, day_of_year),
        'month': np.full(ny * nx, month)
    }
    
    X_grid = pd.DataFrame(grid_data)
    
    # Impute potential NaNs in raw NetCDFs (e.g. cloud gaps) with median
    if X_grid.isna().any().any():
        X_grid = X_grid.fillna(X_grid.median())
        
    predicted_concentrations = {}
    
    for pollutant in TARGET_COLS:
        model_path = os.path.join(models_dir, f"{model_type}_{pollutant}.joblib")
        model = joblib.load(model_path)
        pred = model.predict(X_grid)
        # Reshape back to grid
        predicted_concentrations[pollutant] = np.maximum(pred.reshape(ny, nx), 0.0)
        
    # Calculate gridded CPCB AQI
    # We will do this cell by cell
    aqi_grid = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            row_dict = {pol: float(predicted_concentrations[pol][y, x]) for pol in TARGET_COLS}
            aqi_grid[y, x] = calculate_cpcb_aqi(row_dict)
            
    # Pack into xarray Dataset
    out_ds = xr.Dataset(
        data_vars={
            "PM25": (["lat", "lon"], predicted_concentrations["PM2.5"]),
            "PM10": (["lat", "lon"], predicted_concentrations["PM10"]),
            "NO2": (["lat", "lon"], predicted_concentrations["NO2"]),
            "SO2": (["lat", "lon"], predicted_concentrations["SO2"]),
            "CO": (["lat", "lon"], predicted_concentrations["CO"]),
            "O3": (["lat", "lon"], predicted_concentrations["O3"]),
            "AQI": (["lat", "lon"], aqi_grid)
        },
        coords={
            "lat": ds.lat.values,
            "lon": ds.lon.values,
            "time": pd.to_datetime(dt)
        }
    )
    
    return out_ds

if __name__ == "__main__":
    train_and_evaluate_ml()
