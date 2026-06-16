import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
import folium
import xarray as xr
from datetime import datetime, timedelta
import traceback

# Resolve system paths to load local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml_models import predict_surface_grid
from src.hotspot_detection import process_daily_hotspots
from src.trajectory_analysis import calculate_back_trajectory
from src.utils import (
    get_aqi_color_theme,
    get_station_locations,
    geocode_trajectory_points,
    classify_source_regions,
    summarize_source_regions,
    compute_timeline,
    human_summary
)

# Configure Streamlit page layout
st.set_page_config(
    page_title="Satellite Surface AQI & HCHO Hotspot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium dark-mode styling with glassmorphism cards and custom typography
st.markdown("""
    <style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Title and header styling */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* KPI block */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    .kpi-card {
        flex: 1;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    /* Customize widgets */
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
    }
    .stSlider > div > div > div {
        background-color: #38bdf8 !important;
    }
    
    /* Custom AQI banner */
    .aqi-banner {
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Define directories relative to the repository root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load persistent datasets
@st.cache_data
def load_matched_data():
    df_path = os.path.join(PROCESSED_DIR, "matched_station_data.csv")
    if os.path.exists(df_path):
        return pd.read_csv(df_path)
    return None

@st.cache_data
def load_fire_data():
    fire_path = os.path.join(RAW_DIR, "fire_alerts.csv")
    if os.path.exists(fire_path):
        df = pd.read_csv(fire_path)
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        return df
    return None

@st.cache_data
def load_metrics():
    ml_path = os.path.join(PROCESSED_DIR, "ml_metrics.csv")
    dl_path = os.path.join(PROCESSED_DIR, "dl_metrics.csv")
    ml_df = pd.read_csv(ml_path) if os.path.exists(ml_path) else None
    dl_df = pd.read_csv(dl_path) if os.path.exists(dl_path) else None
    return ml_df, dl_df

@st.cache_data
def load_lag_correlation():
    lag_path = os.path.join(PROCESSED_DIR, "fire_hcho_lag_correlation.csv")
    if os.path.exists(lag_path):
        return pd.read_csv(lag_path)
    return None

# Load caches
df_stations = load_matched_data()
df_fires = load_fire_data()
ml_metrics, dl_metrics = load_metrics()
df_lag = load_lag_correlation()

# Debug helper function
def safe_plotly_chart(fig, container_name="chart", **kwargs):
    """Safely render plotly charts with error handling"""
    try:
        if fig is None:
            st.error(f"❌ {container_name}: Figure is None")
            return False
        st.plotly_chart(fig, **kwargs)
        return True
    except Exception as e:
        st.error(f"❌ {container_name} Error: {str(e)}")
        st.write(f"Debug trace: {traceback.format_exc()}")
        return False


def render_card_header(title, subtitle=None):
    subtitle_html = f"<p style='color: #94a3b8; margin-top: 5px; margin-bottom: 0;'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div class='glass-card' style='padding: 18px 24px 18px 24px;'>"
        f"<h3 style='margin: 0 0 8px 0;'>{title}</h3>"
        f"{subtitle_html}"
        f"</div>",
        unsafe_allow_html=True
    )


def render_html_card(inner_html):
    st.markdown(
        f"<div class='glass-card' style='padding: 24px;'>"
        f"{inner_html}"
        f"</div>",
        unsafe_allow_html=True
    )

# Check if pipeline has run
if df_stations is None:
    st.error("⚠️ Data files not found. Please run the model pipeline first using `python run_pipeline.py` to populate data structures!")
    st.stop()

# Available dates list (retrieve from matched station dates)
available_dates = sorted(df_stations['date'].unique())

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown(
    "<div style='text-align: center; padding: 10px;'><h2 style='color: #38bdf8; margin-bottom: 0;'>ISRO & CPCB</h2>"
    "<p style='color: #64748b; font-size: 0.85rem;'>AQI & HCHO Spatial Intelligence</p></div>", 
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Analysis Module",
    ["🇮🇳 Surface AQI Mapping", "🔥 HCHO Hotspot Detection", "🌬️ Transport & Trajectories", "📊 Performance & Evaluation"]
)

st.sidebar.markdown("---")
selected_date_str = st.sidebar.selectbox("Analysis Date", available_dates)
selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d")

st.sidebar.info(
    "**Project Context**\n\n"
    "This dashboard integrates INSAT-3D AOD, Sentinel-5P column gases, MODIS/VIIRS fire alerts, and ERA5 weather variables to reconstruct surface air quality and trace chemical plumes over India."
)

# --- HEADER TITLE ---
st.markdown(
    f"<div class='glass-card' style='padding: 20px; border-left: 5px solid #38bdf8; margin-bottom: 25px;'>"
    f"<h1 style='margin: 0; font-size: 2.2rem; color: #f8fafc;'>Satellite-Based Surface AQI Estimation & Hotspot Detection</h1>"
    f"<p style='margin: 5px 0 0 0; color: #94a3b8; font-size: 1.05rem;'>Interactive atmospheric analytics system over India | Date: <b>{selected_date_str}</b></p>"
    f"</div>",
    unsafe_allow_html=True
)

# ==========================================
# PAGE 1: SURFACE AQI MAPPING
# ==========================================
if page == "🇮🇳 Surface AQI Mapping":
    st.subheader("Interactive Surface Concentration and CPCB AQI Estimations")
    
    # Model Selection
    model_choice = st.selectbox("Select ML Model Backend for Grid Predictions", ["XGBoost (xgb)", "Random Forest (rf)"])
    model_type = "xgb" if "XGBoost" in model_choice else "rf"
    
    # Run Grid Prediction
    with st.spinner("Reconstructing surface pollution grids..."):
        grid_ds = predict_surface_grid(selected_date_str, MODELS_DIR, RAW_DIR, model_type=model_type)
        
    if grid_ds is None:
        st.warning("Could not load netcdf variables for the selected date. Check simulation files.")
    else:
        # Create KPIs
        max_aqi = int(np.nanmax(grid_ds["AQI"].values))
        mean_pm25 = float(np.nanmean(grid_ds["PM25"].values))
        
        # Count fires for that day
        day_fires = len(df_fires[df_fires['acq_date'] == selected_date_str])
        
        # Display KPIs
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-label">Active Biomass Fires</div>
                    <div class="kpi-value" style="color: #ef4444;">{day_fires}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Max Predicted AQI (India Grid)</div>
                    <div class="kpi-value" style="color: #e11d48;">{max_aqi}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Average PM2.5 Concentration</div>
                    <div class="kpi-value" style="color: #10b981;">{mean_pm25:.2f} µg/m³</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Selected variable to map
        map_var = st.selectbox("Select Variable to Display", ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])
        
        # Map parameters
        var_key = "PM25" if map_var == "PM2.5" else map_var
        data_grid = grid_ds[var_key].values
        
        # Define color scale based on variable
        if map_var == "AQI":
            # CPCB colors
            colorscale = [
                [0.0, "#009966"],  # Good
                [0.1, "#009966"],
                [0.1, "#86c72c"],  # Satisfactory
                [0.2, "#86c72c"],
                [0.2, "#ff9933"],  # Mod Polluted
                [0.4, "#ff9933"],
                [0.4, "#cc0033"],  # Poor
                [0.6, "#cc0033"],
                [0.6, "#660099"],  # Very Poor
                [0.8, "#660099"],
                [0.8, "#7e0023"],  # Severe
                [1.0, "#7e0023"]
            ]
            zmin, zmax = 0, 500
        else:
            colorscale = "Viridis"
            zmin = float(np.nanmin(data_grid))
            zmax = float(np.nanmax(data_grid))
            
        col1, col2 = st.columns([3, 1])
        
        with col1:
            render_card_header(f"Estimated Surface {map_var} Map")
            try:
                # Generate Interactive Plotly Heatmap
                fig = px.imshow(
                    data_grid,
                    x=grid_ds.lon.values,
                    y=grid_ds.lat.values,
                    labels=dict(x="Longitude", y="Latitude", color=map_var),
                    color_continuous_scale=colorscale,
                    range_color=[zmin, zmax],
                    origin='lower',
                    aspect='auto'
                )

                # Map outline formatting
                fig.update_layout(
                    paper_bgcolor='rgba(30,41,59,0.6)',
                    plot_bgcolor='rgba(15,23,42,0.8)',
                    font=dict(color="#e2e8f0", size=12),
                    margin=dict(l=50, r=20, t=30, b=50),
                    height=550,
                    xaxis=dict(title="Longitude (°E)", gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(title="Latitude (°N)", gridcolor="rgba(255,255,255,0.1)"),
                )

                # Plot active fires as coordinates overlay
                if day_fires > 0 and st.checkbox("Overlay MODIS/VIIRS Active Fires"):
                    fires_day = df_fires[df_fires['acq_date'] == selected_date_str]
                    fig.add_trace(
                        go.Scatter(
                            x=fires_day['longitude'],
                            y=fires_day['latitude'],
                            mode='markers',
                            marker=dict(size=4, color='orange', line=dict(width=0.2, color='red')),
                            name='Active Fires'
                        )
                    )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering AQI map: {e}")
                st.write(f"Data shape: {data_grid.shape}, Min: {np.nanmin(data_grid)}, Max: {np.nanmax(data_grid)}")

            # Show CPCB color table
            bands = [
                ("Good (0-50)", "#009966", "Minimal Impact"),
                ("Satisfactory (51-100)", "#86c72c", "Minor breathing discomfort to sensitive people"),
                ("Moderately Polluted (101-200)", "#ff9933", "Breathing discomfort to lungs/heart patients"),
                ("Poor (201-300)", "#cc0033", "Breathing discomfort to most people"),
                ("Very Poor (301-400)", "#660099", "Respiratory illness on prolonged exposure"),
                ("Severe (401-500)", "#7e0023", "Seriously impacts healthy and sensitive populations")
            ]
            
            for band, color, desc in bands:
                st.markdown(
                    f"<div class='aqi-banner' style='background-color: {color}; color: {'#000000' if color in ['#86c72c', '#ff9933'] else '#ffffff'};'>"
                    f"{band}</div>"
                    f"<p style='font-size: 0.85rem; color: #94a3b8; margin-top: -10px; margin-bottom: 15px;'>{desc}</p>",
                    unsafe_allow_html=True
                )

# ==========================================
# PAGE 2: HCHO HOTSPOT DETECTION
# ==========================================
elif page == "🔥 HCHO Hotspot Detection":
    st.subheader("Sentinel-5P HCHO Column Densities & Biomass Burning Co-Analysis")
    
    # Load daily hotspots
    with st.spinner("Analyzing HCHO hotspot statistics..."):
        hs_ds = process_daily_hotspots(selected_date_str, RAW_DIR, PROCESSED_DIR)
        
    if hs_ds is None:
        st.warning("Gridded HCHO dataset not available.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_card_header("HCHO Hotspot Cluster Map")
            
            try:
                # Select hotspot detection method
                method = st.radio("Hotspot Detection Algorithm", ["Getis-Ord Gi* z-score", "DBSCAN Spatial Clustering", "Statistical Thresholding (> mean + 2σ)"], horizontal=True)
                
                if method == "Getis-Ord Gi* z-score":
                    grid_data = hs_ds["Getis_Ord_Z"].values
                    cmap = "RdBu_r"
                    zmin, zmax = -3, 3
                    label_name = "Gi* z-score"
                elif method == "DBSCAN Spatial Clustering":
                    grid_data = hs_ds["DBSCAN_Clusters"].values
                    # We want cluster colors
                    cmap = "plotly3"
                    zmin, zmax = -2, 5
                    label_name = "Cluster ID (-1=Noise, -2=BG)"
                else:
                    grid_data = hs_ds["Threshold_Hotspots"].values
                    cmap = "Reds"
                    zmin, zmax = 0, 1
                    label_name = "Hotspot Pixel"
                    
                fig = px.imshow(
                    grid_data,
                    x=hs_ds.lon.values,
                    y=hs_ds.lat.values,
                    labels=dict(x="Longitude", y="Latitude", color=label_name),
                    color_continuous_scale=cmap,
                    range_color=[zmin, zmax],
                    origin='lower',
                    aspect='auto'
                )
                
                fig.update_layout(
                    paper_bgcolor='rgba(30,41,59,0.6)',
                    plot_bgcolor='rgba(15,23,42,0.8)',
                    font=dict(color="#e2e8f0", size=12),
                    margin=dict(l=50, r=20, t=30, b=50),
                    height=500,
                    xaxis=dict(title="Longitude (°E)", gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(title="Latitude (°N)", gridcolor="rgba(255,255,255,0.1)"),
                )
                
                # Overlay fires
                fires_day = df_fires[df_fires['acq_date'] == selected_date_str]
                if len(fires_day) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=fires_day['longitude'],
                            y=fires_day['latitude'],
                            mode='markers',
                            marker=dict(size=fires_day['frp']/12.0 + 2, color='rgba(239, 68, 68, 0.6)'),
                            name='Active Fires'
                        )
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering hotspot map: {e}")
                st.write(f"Traceback: {traceback.format_exc()}")
            
        with col2:
            render_card_header("HCHO-Fire Lag Correlation")
            
            try:
                # Show lag table
                if df_lag is not None:
                    st.table(df_lag)
                    
                    # Plot lag correlation
                    fig_lag = px.bar(
                        df_lag,
                        x="Lag_Days",
                        y="Correlation",
                        text="Correlation",
                        labels={"Lag_Days": "Lag Days (Fires lead HCHO)", "Correlation": "Pearson Correlation (r)"},
                        title="Correlation Strength by Temporal Lag"
                    )
                    fig_lag.update_layout(
                        paper_bgcolor='rgba(30,41,59,0.6)',
                        plot_bgcolor='rgba(15,23,42,0.8)',
                        font=dict(color="#e2e8f0", size=12),
                        margin=dict(l=50, r=20, t=40, b=40),
                        height=280
                    )
                    fig_lag.update_traces(marker_color='#38bdf8', textposition='outside')
                    st.plotly_chart(fig_lag, use_container_width=True)
                else:
                    st.write("Lag correlation metrics not found.")
            except Exception as e:
                st.error(f"Error rendering lag correlation chart: {e}")
                st.write(f"Traceback: {traceback.format_exc()}")

# ==========================================
# PAGE 3: WIND & TRAJECTORY TRANSPORT
# ==========================================
elif page == "🌬️ Transport & Trajectories":
    st.subheader("Atmospheric Transport Pathway and Kinematic Back-Trajectories")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_card_header("Back-Trajectory Controls")
        
        # Selected City Coordinates
        cities = {
            "New Delhi (IGP Receptor)": (28.61, 77.20),
            "Patna (Downwind IGP)": (25.59, 85.13),
            "Amritsar (Source Region)": (31.63, 74.87),
            "Mumbai (West Coast)": (19.07, 72.87),
            "Bengaluru (South Peninsula)": (12.97, 77.59),
            "Kolkata (East Coast)": (22.57, 88.36)
        }
        
        sel_city = st.selectbox("Select Receptor Location", list(cities.keys()))
        city_coords = cities[sel_city]
        
        duration = st.slider("Trajectory Backwards Duration (Hours)", 24, 72, 48, step=12)
        
        with st.spinner("Calculating kinematic back-trajectory..."):
            traj_df = calculate_back_trajectory(
                city_coords[0], city_coords[1], 
                selected_date_str, RAW_DIR, 
                duration_hours=duration
            )
        
        st.success(f"Successfully calculated {len(traj_df)} backward steps.")
        
        st.write("#### Kinematic Path Details")
        st.dataframe(traj_df[['hour', 'latitude', 'longitude', 'u10', 'v10']].head(10))
        
        if not traj_df.empty:
            traj_enriched = geocode_trajectory_points(traj_df)
            source_df = classify_source_regions(traj_enriched, df_fires, RAW_DIR)
            source_summary = summarize_source_regions(source_df, duration)
            timeline = compute_timeline(traj_enriched)
            summary_text = human_summary(
                source_summary['primary'],
                source_summary['secondary'],
                source_summary['confidence'],
                source_summary['duration']
            )
            
            render_card_header("Likely Source Regions")
            st.markdown(f"<p><strong>Primary Source Region:</strong> {source_summary['primary']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Secondary Source Region:</strong> {source_summary['secondary']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Transport Duration:</strong> {source_summary['duration']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Confidence:</strong> {source_summary['confidence']}</p>", unsafe_allow_html=True)
            
            render_card_header("Region Attribution")
            st.dataframe(traj_enriched[['hour', 'latitude', 'longitude', 'place', 'district', 'state', 'country']].head(10))
    
    with col2:
        render_card_header("Trajectory Overlay Map")
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c
        
        # Set map center at receptor city
        map_center = [city_coords[0], city_coords[1]]
        m = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB dark_matter")
        
        # Plot receptor city marker
        folium.Marker(
            location=map_center,
            popup=f"Receptor: {sel_city}",
            icon=folium.Icon(color="darkred", icon="info-sign")
        ).add_to(m)
        
        # Plot trajectory line
        points = list(zip(traj_df['latitude'], traj_df['longitude']))
        folium.PolyLine(
            points,
            color="red",
            weight=3,
            opacity=0.8,
            tooltip=f"{duration}h Back-Trajectory from {sel_city}"
        ).add_to(m)
        
        # Add circular markers for 12h intervals
        for idx, row in traj_df.iterrows():
            hr = int(row['hour'])
            if hr in [-12, -24, -36, -48, -72]:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    color="cyan",
                    fill=True,
                    fill_color="blue",
                    popup=f"{hr} hours ago",
                ).add_to(m)
        
        # Overlay active fires near trajectory path (within 1 degree of any point on path)
        fires_sub = df_fires[df_fires['acq_date'] == selected_date_str]
        if len(fires_sub) > 0:
            for idx, fire in fires_sub.iterrows():
                dists = np.sqrt((traj_df['latitude'] - fire['latitude'])**2 + (traj_df['longitude'] - fire['longitude'])**2)
                if np.min(dists) < 1.5:
                    folium.CircleMarker(
                        location=[fire['latitude'], fire['longitude']],
                        radius=2 + fire['frp']/100.0,
                        color="orange",
                        fill=True,
                        fill_color="red",
                        opacity=0.5,
                        popup=f"FRP: {fire['frp']}"
                    ).add_to(m)
        
        if not traj_df.empty:
            region_centroids = {
                'Delhi NCR': (28.6, 77.2),
                'Punjab Agricultural Belt': (30.5, 75.8),
                'Ludhiana Industrial Region': (30.9, 75.8),
                'Panipat Industrial Cluster': (29.4, 76.9),
                'Rural Area': (28.5, 76.0)
            }
            source_groups = source_df.groupby('region').first().reset_index()
            for _, src in source_groups.iterrows():
                if src['region'] in region_centroids:
                    coord = region_centroids[src['region']]
                    distance = haversine(map_center[0], map_center[1], coord[0], coord[1])
                    folium.Marker(
                        location=list(coord),
                        popup=(f"<b>{src['region']}</b><br>Type: {src['category']}<br>"
                               f"Confidence: {int(src['confidence'])}%<br>"
                               f"Distance: {distance:.1f} km"),
                        icon=folium.Icon(color="lightblue", icon="map-marker")
                    ).add_to(m)
        
        st_folium(m, width="100%", height=550)
        
    if not traj_df.empty:
        render_card_header("Source Region Timeline")
        for step in timeline:
            st.markdown(f"**{step['label']}** → {step['region']}")
        render_card_header("Contribution Ranking")
        st.table(source_summary['ranking'])
        
        render_card_header("Probabilistic Source Attribution")
        st.markdown(
            "Source attribution is probabilistic and based on atmospheric transport modeling, satellite observations, and meteorological conditions. "
            "Results indicate likely contributing regions and should not be interpreted as exact emission sources.",
            unsafe_allow_html=True
        )
        st.markdown(f"<p>{summary_text}</p>", unsafe_allow_html=True)

# ==========================================
# PAGE 4: PERFORMANCE & EVALUATION
# ==========================================
elif page == "📊 Performance & Evaluation":

    st.subheader("Model Performance Validation and Variable Importance Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_card_header("Machine Learning Model Comparison")
        
        st.write(f"**DEBUG:** ml_metrics is None: {ml_metrics is None}")
        st.write(f"**DEBUG:** ml_metrics type: {type(ml_metrics)}")
        
        try:
            # Display ML Metrics Table
            if ml_metrics is not None:
                st.write("#### Validation Metrics (Random Forest & XGBoost)")
                st.dataframe(ml_metrics)
                st.write("**Attempting to render bar chart...**")
                
                # Plot metrics comparison
                fig_met = px.bar(
                    ml_metrics,
                    x="pollutant",
                    y="R2",
                    color="model",
                    barmode="group",
                    labels={"pollutant": "Target Pollutant", "R2": "R² (Coefficient of Determination)"},
                    title="R² Score Comparison by Pollutant & Model"
                )
                fig_met.update_layout(
                    paper_bgcolor='rgba(30,41,59,0.6)',
                    plot_bgcolor='rgba(15,23,42,0.8)',
                    font=dict(color="#e2e8f0", size=12),
                    margin=dict(l=50, r=20, t=40, b=40),
                    height=350,
                    showlegend=True,
                    xaxis_title="Target Pollutant",
                    yaxis_title="R² Score"
                )
                safe_plotly_chart(fig_met, container_name="ML metrics chart", use_container_width=True)
            else:
                st.warning("ML metrics file not found.")
        except Exception as e:
            st.error(f"❌ Error rendering ML metrics chart: {str(e)}")
            st.write(f"Full error: {traceback.format_exc()}")
        
    with col2:
        render_card_header("Deep Learning (CNN-LSTM) Performance")
        
        st.write(f"**DEBUG:** dl_metrics is None: {dl_metrics is None}")
        st.write(f"**DEBUG:** dl_metrics type: {type(dl_metrics)}")
        
        try:
            # Display DL Metrics Table
            if dl_metrics is not None:
                st.write("#### Validation Metrics (CNN-LSTM)")
                st.dataframe(dl_metrics)
                st.write("**Attempting to render bar chart...**")
                
                # Plot correlation coefficients
                fig_dl = px.bar(
                    dl_metrics,
                    x="pollutant",
                    y="Pearson_R",
                    labels={"pollutant": "Target Pollutant", "Pearson_R": "Pearson Correlation (r)"},
                    title="CNN-LSTM Pearson Correlation (r)"
                )
                fig_dl.update_layout(
                    paper_bgcolor='rgba(30,41,59,0.6)',
                    plot_bgcolor='rgba(15,23,42,0.8)',
                    font=dict(color="#e2e8f0", size=12),
                    margin=dict(l=50, r=20, t=40, b=40),
                    height=350,
                    xaxis_title="Target Pollutant",
                    yaxis_title="Pearson Correlation (r)"
                )
                fig_dl.update_traces(marker_color="#10b981")
                safe_plotly_chart(fig_dl, container_name="DL metrics chart", use_container_width=True)
            else:
                st.warning("CNN-LSTM metrics file not found.")
        except Exception as e:
            st.error(f"❌ Error rendering DL metrics chart: {str(e)}")
            st.write(f"Full error: {traceback.format_exc()}")
