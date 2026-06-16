import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Computes standard evaluation metrics for regression tasks:
    RMSE, MAE, R2, and Pearson correlation coefficient.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter out NaNs if present
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if not np.any(mask):
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "Pearson_R": np.nan}
        
    yt = y_true[mask]
    yp = y_pred[mask]
    
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    
    # Calculate Pearson correlation coefficient
    if len(np.unique(yp)) > 1 and len(np.unique(yt)) > 1:
        r_coeff, p_val = pearsonr(yt, yp)
    else:
        r_coeff, p_val = 0.0, 1.0
        
    return {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "R2": round(r2, 3),
        "Pearson_R": round(r_coeff, 3),
        "P_Value": round(p_val, 6)
    }

def get_aqi_color_theme(aqi):
    """
    Returns hex codes and textual descriptions for standard CPCB bands.
    """
    if pd.isna(aqi) or aqi < 0:
        return {"category": "No Data", "color": "#7f8c8d", "text_color": "#ffffff"}
        
    if aqi <= 50:
        return {"category": "Good", "color": "#009966", "text_color": "#ffffff"}
    elif aqi <= 100:
        return {"category": "Satisfactory", "color": "#86c72c", "text_color": "#000000"}
    elif aqi <= 200:
        return {"category": "Moderately Polluted", "color": "#ff9933", "text_color": "#000000"}
    elif aqi <= 300:
        return {"category": "Poor", "color": "#cc0033", "text_color": "#ffffff"}
    elif aqi <= 400:
        return {"category": "Very Poor", "color": "#660099", "text_color": "#ffffff"}
    else:
        return {"category": "Severe", "color": "#7e0023", "text_color": "#ffffff"}

def get_station_locations(base_dir="c:/Users/Anjali/OneDrive/Desktop/ISRO"):
    """
    Returns unique station IDs, names, and coordinates from simulated ground data.
    """
    raw_path = f"{base_dir}/data/raw/cpcb_stations.csv"
    if not os.path.exists(raw_path):
        return []
    df = pd.read_csv(raw_path)
    stations = df[['station_id', 'station_name', 'latitude', 'longitude']].drop_duplicates().to_dict('records')
    return stations


def geocode_trajectory_points(df):
    """Reverse geocode trajectory points to region names.

    This is a simple local lookup using an internal region grid or
    shapefile-like approximation from available data.
    """
    # Use a lightweight reverse geocoding lookup table if available.
    # For offline mode we approximate using very coarse administrative boundaries.
    region_lookup = [
        {'lat_min': 28.0, 'lat_max': 30.5, 'lon_min': 74.0, 'lon_max': 77.5,
         'place': 'Panipat', 'district': 'Panipat', 'state': 'Haryana', 'country': 'India'},
        {'lat_min': 29.5, 'lat_max': 31.9, 'lon_min': 75.5, 'lon_max': 76.7,
         'place': 'Ludhiana', 'district': 'Ludhiana', 'state': 'Punjab', 'country': 'India'},
        {'lat_min': 30.0, 'lat_max': 32.2, 'lon_min': 75.8, 'lon_max': 78.6,
         'place': 'Chandigarh', 'district': 'Chandigarh', 'state': 'Punjab', 'country': 'India'},
        {'lat_min': 26.0, 'lat_max': 28.2, 'lon_min': 83.0, 'lon_max': 86.0,
         'place': 'Patna', 'district': 'Patna', 'state': 'Bihar', 'country': 'India'},
        {'lat_min': 19.0, 'lat_max': 21.5, 'lon_min': 72.0, 'lon_max': 74.5,
         'place': 'Mumbai', 'district': 'Mumbai', 'state': 'Maharashtra', 'country': 'India'},
        {'lat_min': 22.5, 'lat_max': 23.5, 'lon_min': 88.0, 'lon_max': 89.0,
         'place': 'Kolkata', 'district': 'Kolkata', 'state': 'West Bengal', 'country': 'India'},
    ]

    def lookup(lat, lon):
        for region in region_lookup:
            if region['lat_min'] <= lat <= region['lat_max'] and region['lon_min'] <= lon <= region['lon_max']:
                return region
        return {'place': 'Rural Area', 'district': 'Unknown', 'state': 'Unknown', 'country': 'India'}

    enriched = []
    for _, row in df.iterrows():
        region = lookup(row['latitude'], row['longitude'])
        enriched.append({
            **row.to_dict(),
            'place': region['place'],
            'district': region['district'],
            'state': region['state'],
            'country': region['country']
        })

    return pd.DataFrame(enriched)


def classify_source_regions(df_traj, fire_df, raw_dir):
    """Classify source contributions for each trajectory point."""
    sources = []

    # helper functions
    def is_fire_point(lat, lon):
        nearby = fire_df[(np.abs(fire_df['latitude'] - lat) < 0.5) & (np.abs(fire_df['longitude'] - lon) < 0.5)]
        return len(nearby) > 0, nearby

    def is_pollution_zone(lat, lon):
        if 74.0 <= lon <= 77.5 and 28.0 <= lat <= 31.5:
            return 'Delhi NCR', 0.6
        if 74.0 <= lon <= 76.8 and 29.5 <= lat <= 31.5:
            return 'Punjab Agricultural Belt', 0.8
        if 75.0 <= lon <= 76.5 and 29.0 <= lat <= 31.0:
            return 'Ludhiana Industrial Region', 0.7
        if 28.5 <= lon <= 77.5 and 29.0 <= lat <= 30.5:
            return 'Panipat Industrial Cluster', 0.65
        return None, 0.0

    for _, point in df_traj.iterrows():
        region_name, base_score = is_pollution_zone(point['latitude'], point['longitude'])
        fire_flag, nearby_fires = is_fire_point(point['latitude'], point['longitude'])
        if region_name is None:
            region_name = point['place']
            base_score = 0.35

        score = base_score
        contribution = 'Background Transport'
        if fire_flag:
            score = min(1.0, score + 0.25)
            contribution = 'Crop Residue Burning'
        if region_name == 'Delhi NCR':
            contribution = 'Vehicular Emissions'
        elif region_name == 'Ludhiana Industrial Region' or region_name == 'Panipat Industrial Cluster':
            contribution = 'Industrial Emissions'
        elif region_name == 'Punjab Agricultural Belt':
            contribution = 'Crop Residue Burning'

        sources.append({
            'region': region_name,
            'category': contribution,
            'confidence': round(score * 100, 0),
            'fire_points': len(nearby_fires),
            'avg_u10': point['u10'],
            'avg_v10': point['v10']
        })

    return pd.DataFrame(sources)


def summarize_source_regions(source_df, duration_hours):
    """Create a short source summary and ranking."""
    if source_df is None or source_df.empty:
        return {
            'primary': 'Unknown',
            'secondary': 'Unknown',
            'duration': f"{duration_hours} Hours",
            'confidence': 'N/A',
            'ranking': []
        }

    summary = source_df.groupby(['region', 'category']).agg(
        score_mean=('confidence', 'mean'),
        points=('region', 'count'),
        fire_points=('fire_points', 'sum')
    ).reset_index()
    summary['rank_score'] = summary['score_mean'] * 0.6 + (summary['points'] / summary['points'].max()) * 20 + (summary['fire_points'] / (summary['fire_points'].max() or 1)) * 10
    summary = summary.sort_values(['rank_score', 'score_mean'], ascending=False)

    primary = summary.iloc[0]
    secondary = summary.iloc[1] if len(summary) > 1 else primary

    ranking = summary[['region', 'category', 'score_mean']].rename(columns={'score_mean': 'Contribution Score'})
    ranking['Contribution Score'] = ranking['Contribution Score'].round(1).astype(str) + '%'

    return {
        'primary': primary['region'],
        'secondary': secondary['region'],
        'duration': f"{duration_hours} Hours",
        'confidence': f"{round(primary['score_mean'], 0)}%",
        'ranking': ranking
    }


def compute_timeline(df_traj):
    timeline = []
    for _, row in df_traj.iterrows():
        if row['hour'] in [-48, -36, -24, -12, 0]:
            timeline.append({
                'label': f"{int(row['hour'] * -1)} Hours Ago" if row['hour'] != 0 else 'Current',
                'region': f"{row['place']} ({row['district']}, {row['state']})"
            })
    return timeline


def human_summary(primary, secondary, confidence, duration):
    return (
        f"Analysis indicates that the air mass reaching the receptor on the selected date primarily originated from {primary} "
        f"and passed through secondary regions including {secondary}. Active fire hotspots detected along the transport pathway suggest crop residue burning as the dominant contributor, with secondary contributions from industrial emissions. "
        f"Transport duration: {duration}. Confidence in the primary source attribution is {confidence}."
    )
