import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

# Feature names and target names matching ML models
FEATURE_COLS = [
    'AOD', 'col_NO2', 'col_SO2', 'col_CO', 'col_O3', 'col_HCHO', 
    't2m', 'r2', 'blh', 'u10', 'v10'
]
TARGET_COLS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

class CNNLSTM(nn.Module):
    def __init__(self, num_features, seq_len, patch_size, num_targets=6):
        super(CNNLSTM, self).__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        
        # CNN component: extracts spatial context from a patch
        # Input size per step: (batch_size, num_features, patch_size, patch_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), # output size: 2x2
            nn.Flatten() # size: 32 * 2 * 2 = 128
        )
        
        cnn_out_dim = 128
        
        # LSTM component: processes temporal sequence of spatial features
        # Input shape: (batch_size, seq_len, cnn_out_dim)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True
        )
        
        # Regressor Head
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_targets)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features, patch_size, patch_size)
        batch_size = x.size(0)
        
        # Reshape to pass through CNN: (batch_size * seq_len, num_features, patch_size, patch_size)
        x_reshaped = x.view(batch_size * self.seq_len, -1, self.patch_size, self.patch_size)
        spatial_features = self.cnn(x_reshaped)
        
        # Reshape back to sequence: (batch_size, seq_len, cnn_out_dim)
        seq_features = spatial_features.view(batch_size, self.seq_len, -1)
        
        # LSTM outputs
        lstm_out, (hn, cn) = self.lstm(seq_features)
        
        # Take the last time step output: (batch_size, hidden_size)
        last_step = lstm_out[:, -1, :]
        
        # Predict targets
        out = self.fc(last_step)
        return out

class StationPatchDataset(Dataset):
    def __init__(self, samples):
        """
        samples: list of dicts, each with 'x' (numpy array shape [seq_len, num_features, H, W]) and 'y' (target array shape [6])
        """
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx]['x'], dtype=torch.float32)
        y = torch.tensor(self.samples[idx]['y'], dtype=torch.float32)
        return x, y

def build_dataset_samples(base_dir, seq_len=7, patch_size=5):
    """
    Reads matched CSV and gridded NetCDFs, pre-extracts temporal spatial patches
    surrounding stations, and groups them into sequential samples.
    """
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    df = pd.read_csv(os.path.join(processed_dir, "matched_station_data.csv"))
    df['date'] = pd.to_datetime(df['date'])
    
    # Load all gridded files into memory to enable fast patch extraction
    unique_dates = sorted(df['date'].dt.strftime("%Y-%m-%d").unique())
    grid_cache = {}
    
    from src.preprocessing import load_daily_netcdf
    for date_str in unique_dates:
        ds = load_daily_netcdf(date_str, raw_dir)
        if ds is not None:
            # Stack features along new coordinate
            feat_arrays = []
            for col in FEATURE_COLS:
                var_name = col.replace("col_", "")
                if var_name == "AOD":
                    feat_arrays.append(ds["AOD"].values)
                else:
                    feat_arrays.append(ds[var_name].values)
            stacked = np.stack(feat_arrays, axis=0) # shape: [num_features, lat, lon]
            grid_cache[date_str] = {
                'data': stacked,
                'lats': ds.lat.values,
                'lons': ds.lon.values
            }
            
    samples = []
    
    # Iterate over unique stations and construct timelines
    for station_id in df['station_id'].unique():
        stn_df = df[df['station_id'] == station_id].sort_values('date').reset_index(drop=True)
        
        for i in range(seq_len - 1, len(stn_df)):
            target_row = stn_df.iloc[i]
            target_date = target_row['date']
            
            # Retrieve sequence of grids
            seq_grids = []
            valid_seq = True
            
            for offset in range(seq_len - 1, -1, -1):
                hist_date = target_date - timedelta(days=offset)
                hist_date_str = hist_date.strftime("%Y-%m-%d")
                
                if hist_date_str not in grid_cache:
                    valid_seq = False
                    break
                    
                grid_info = grid_cache[hist_date_str]
                stacked_grid = grid_info['data']
                lats = grid_info['lats']
                lons = grid_info['lons']
                
                # Get indices for station coordinates
                lat_idx = np.argmin(np.abs(lats - target_row['latitude']))
                lon_idx = np.argmin(np.abs(lons - target_row['longitude']))
                
                # Slice patch
                r = patch_size // 2
                lat_start = max(lat_idx - r, 0)
                lat_end = min(lat_idx + r + 1, len(lats))
                lon_start = max(lon_idx - r, 0)
                lon_end = min(lon_idx + r + 1, len(lons))
                
                patch = stacked_grid[:, lat_start:lat_end, lon_start:lon_end]
                
                # Pad if patch is smaller than patch_size due to boundaries
                if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                    padded_patch = np.zeros((stacked_grid.shape[0], patch_size, patch_size))
                    h_slice = slice(0, patch.shape[1])
                    w_slice = slice(0, patch.shape[2])
                    padded_patch[:, h_slice, w_slice] = patch
                    patch = padded_patch
                    
                seq_grids.append(patch)
                
            if valid_seq:
                x_seq = np.stack(seq_grids, axis=0) # shape: [seq_len, num_features, H, W]
                y_val = target_row[TARGET_COLS].values.astype(np.float32)
                
                samples.append({
                    'x': x_seq,
                    'y': y_val,
                    'date': target_date,
                    'station_id': station_id
                })
                
    return samples

def train_and_evaluate_dl(base_dir="c:/Users/Anjali/OneDrive/Desktop/ISRO", epochs=20, batch_size=32):
    print("Preparing CNN-LSTM training data...")
    samples = build_dataset_samples(base_dir)
    
    if len(samples) == 0:
        raise ValueError("No valid sequential samples built. Verify data simulator dates.")
        
    # Split samples temporally: Train before 2025-12-01, Validate from 2025-12-01 onwards
    split_date = datetime(2025, 12, 1)
    
    train_samples = [s for s in samples if s['date'] < split_date]
    test_samples = [s for s in samples if s['date'] >= split_date]
    
    print(f"DL Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")
    
    train_dataset = StationPatchDataset(train_samples)
    test_dataset = StationPatchDataset(test_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = CNNLSTM(num_features=len(FEATURE_COLS), seq_len=7, patch_size=5, num_targets=6).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
            
        epoch_loss /= len(train_dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")
            
    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            predictions = model(x_batch)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())
            
    preds_arr = np.concatenate(all_preds, axis=0) # shape [N, 6]
    targets_arr = np.concatenate(all_targets, axis=0) # shape [N, 6]
    
    # Save Model Weights
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "cnn_lstm_pollutants.pth"))
    
    # Calculate metrics per pollutant
    metrics_summary = []
    for idx, pollutant in enumerate(TARGET_COLS):
        y_test = targets_arr[:, idx]
        pred = np.maximum(preds_arr[:, idx], 0.0) # physical constraint
        
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        
        # Pearson correlation
        if len(np.unique(pred)) > 1 and len(np.unique(y_test)) > 1:
            r, _ = pearsonr(y_test, pred)
        else:
            r = 0.0
            
        metrics_summary.append({
            "pollutant": pollutant,
            "model": "CNN-LSTM",
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
            "R2": round(r2, 3),
            "Pearson_R": round(r, 3)
        })
        
    metrics_df = pd.DataFrame(metrics_summary)
    processed_dir = os.path.join(base_dir, "data", "processed")
    metrics_df.to_csv(os.path.join(processed_dir, "dl_metrics.csv"), index=False)
    
    print("\n--- Deep Learning (CNN-LSTM) Validation Metrics ---")
    print(metrics_df.to_string(index=False))
    
    return metrics_df

if __name__ == "__main__":
    train_and_evaluate_dl()
