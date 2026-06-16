# ISRO Air Quality & HCHO Dashboard

This project is a Streamlit-based dashboard for air quality estimation, HCHO hotspot detection, and transport analysis using simulated satellite and ground station data.

## Deployment

### Requirements

- Python 3.11+
- `pip install -r requirements.txt`

### Run locally

```bash
cd ISRO
python run_pipeline.py
streamlit run dashboard/app.py
```

### Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud.
3. Create a new app and connect it to the repository.
4. Set the main file path to `dashboard/app.py`.
5. Use default branch `main`.

### Notes

- The app reads from `data/raw/` and `data/processed/`.
- Make sure preprocessed files like `data/processed/matched_station_data.csv` exist before opening the dashboard.
- Large raw NetCDF files should not be modified after deployment.
