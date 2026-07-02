🌍 Air Quality & HCHO Spatial Intelligence Dashboard
📌 Overview

This project is an AI-powered geospatial dashboard developed to monitor air quality, detect HCHO (Formaldehyde) hotspots, analyze pollution transport pathways, and predict pollutant concentrations using Machine Learning models. The system combines satellite observations, meteorological data, and fire hotspot information to provide actionable environmental insights through interactive visualizations.

🚀 Features
🌫️ Air Quality Monitoring
AQI visualization and pollutant analysis
Monitoring of PM2.5, PM10, NO₂, SO₂, CO, and O₃
Interactive charts and spatial mapping
🔥 HCHO Hotspot Detection
Identification of high formaldehyde concentration regions
Spatial hotspot visualization
Environmental risk assessment
🌬️ Atmospheric Transport Analysis
Backward air-mass trajectory modeling
Pollution transport pathway visualization
Source region identification
Wind vector analysis using meteorological data
🤖 Machine Learning Prediction
Random Forest
XGBoost
CNN-LSTM

Model performance is evaluated using:

RMSE
MAE
R² Score
Pearson Correlation
🗺️ Interactive Dashboard
User-friendly interface built with Streamlit
Geospatial visualizations
Performance comparison dashboards
Real-time analytical insights
📊 Data Sources

The project integrates multiple environmental datasets:

INSAT-3D Atmospheric Data
Sentinel-5P Satellite Observations
MODIS Fire Data
VIIRS Fire Hotspots
ERA5 Meteorological Data
Air Quality Monitoring Data
🛠️ Tech Stack
Python
Streamlit
Pandas
NumPy
Plotly
Scikit-learn
TensorFlow/Keras
GIS & Remote Sensing Technologies
🎯 Project Objective

The main objective of this project is to provide a unified platform for:

Monitoring air quality conditions
Detecting HCHO hotspots
Understanding pollution transport mechanisms
Supporting environmental decision-making
Leveraging AI for pollutant prediction and analysis
📈 Future Improvements
Real-time data integration
Advanced source attribution system
District-wise pollution contribution analysis
AI-generated environmental reports
Pollution forecasting and early warning system

## Official Data Ingestion Pipeline
The project now includes official-source ingestion and preprocessing scripts to build a production-ready data engineering pipeline.

Key scripts:
- `run_pipeline.py` — orchestrates simulation or real-mode pipeline execution
- `src/real_data_ingestion/download_all.py` — downloads CPCB, FIRMS, ERA5, Sentinel-5P, MOSDAC sources and preprocesses raw files
- `src/download_cpcb.py` — downloads CPCB station monitoring data
- `src/download_firms.py` — downloads NASA FIRMS active fire alerts
- `src/download_meteorology.py` — downloads ERA5 meteorology via CDS
- `src/download_sentinel5p.py` — ingests Sentinel-5P via Google Earth Engine
- `src/real_data_ingestion/download_mosdac.py` — converts MOSDAC INSAT-3D HDF5 to NetCDF
- `src/preprocess_data.py` — standardizes CPCB and fire alert CSVs
- `src/merge_datasets.py` — merges daily raw grids into a single NetCDF file

Output files:
- `data/raw/cpcb_stations.csv`
- `data/raw/fire_alerts.csv`
- `data/raw/insat3d/*.nc`
- `data/raw/s5p/*.nc`
- `data/raw/era5/*.nc`
- `data/processed/matched_station_data.csv`
- `data/processed/fire_hcho_lag_correlation.csv`
- `data/processed/merged_daily_grids.nc`

Example command for real data ingestion:
```bash
python run_pipeline.py --mode real --download --start-date 2025-10-01 --end-date 2025-10-05 \
  --firms-api-key YOUR_FIRMS_KEY \
  --cpcb-url "https://example.com/cpcb_stations.csv" \
  --mosdac-h5 "path/to/mosdac_files" \
  --skip-existing
```
