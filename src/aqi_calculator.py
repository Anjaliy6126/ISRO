import numpy as np
import pandas as pd

def calculate_sub_index(concentration, breakpoints, index_breakpoints):
    """
    Applies the linear interpolation formula to calculate the AQI sub-index:
    I = I_low + ((I_high - I_low) / (C_high - C_low)) * (C - C_low)
    """
    if pd.isna(concentration) or concentration < 0:
        return np.nan
        
    for i in range(len(breakpoints)):
        c_low, c_high = breakpoints[i]
        i_low, i_high = index_breakpoints[i]
        
        if c_low <= concentration <= c_high:
            return i_low + ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low)
            
    # If concentration exceeds the maximum breakpoint, return the highest index value (capped at 500)
    return 500.0

def get_pm25_sub_index(pm25):
    breakpoints = [(0, 30), (30.1, 60), (60.1, 90), (90.1, 120), (120.1, 250), (250.1, 380)]
    index_breakpoints = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    return calculate_sub_index(pm25, breakpoints, index_breakpoints)

def get_pm10_sub_index(pm10):
    breakpoints = [(0, 50), (50.1, 100), (100.1, 250), (250.1, 350), (350.1, 430), (430.1, 510)]
    index_breakpoints = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    return calculate_sub_index(pm10, breakpoints, index_breakpoints)

def get_no2_sub_index(no2):
    breakpoints = [(0, 40), (40.1, 80), (80.1, 180), (180.1, 280), (280.1, 400), (400.1, 500)]
    index_breakpoints = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    return calculate_sub_index(no2, breakpoints, index_breakpoints)

def get_so2_sub_index(so2):
    breakpoints = [(0, 40), (40.1, 80), (80.1, 380), (380.1, 800), (800.1, 1600), (1600.1, 1800)]
    index_breakpoints = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    return calculate_sub_index(so2, breakpoints, index_breakpoints)

def get_co_sub_index(co):
    # CO is measured in mg/m3
    breakpoints = [(0, 1.0), (1.01, 2.0), (2.01, 10.0), (10.01, 17.0), (17.01, 34.0), (34.01, 50.0)]
    index_breakpoints = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    return calculate_sub_index(co, breakpoints, index_breakpoints)

def get_o3_sub_index(o3):
    breakpoints = [(0, 50), (50.1, 100), (100.1, 168), (168.1, 208), (208.1, 748), (748.1, 1000)]
    index_breakpoints = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    return calculate_sub_index(o3, breakpoints, index_breakpoints)

def calculate_cpcb_aqi(row):
    """
    Calculates overall CPCB AQI from a row or dict containing concentrations of
    PM2.5, PM10, NO2, SO2, CO, and O3.
    CPCB standard requirements:
    1. At least three sub-indices must be present.
    2. One of the three must be either PM2.5 or PM10.
    """
    sub_indices = {}
    
    if "PM2.5" in row and not pd.isna(row["PM2.5"]):
        sub_indices["PM2.5"] = get_pm25_sub_index(row["PM2.5"])
    if "PM10" in row and not pd.isna(row["PM10"]):
        sub_indices["PM10"] = get_pm10_sub_index(row["PM10"])
    if "NO2" in row and not pd.isna(row["NO2"]):
        sub_indices["NO2"] = get_no2_sub_index(row["NO2"])
    if "SO2" in row and not pd.isna(row["SO2"]):
        sub_indices["SO2"] = get_so2_sub_index(row["SO2"])
    if "CO" in row and not pd.isna(row["CO"]):
        sub_indices["CO"] = get_co_sub_index(row["CO"])
    if "O3" in row and not pd.isna(row["O3"]):
        sub_indices["O3"] = get_o3_sub_index(row["O3"])
        
    valid_sub_indices = {k: v for k, v in sub_indices.items() if not pd.isna(v)}
    
    # Check conditions
    if len(valid_sub_indices) < 3:
        return np.nan
        
    if "PM2.5" not in valid_sub_indices and "PM10" not in valid_sub_indices:
        return np.nan
        
    # AQI is the maximum of individual sub-indices
    aqi = max(valid_sub_indices.values())
    return round(aqi)

def get_aqi_category(aqi):
    """
    Returns the CPCB air quality category name, color, and description based on AQI value.
    """
    if pd.isna(aqi) or aqi < 0:
        return "Unknown", "#808080", "No Data"
        
    if aqi <= 50:
        return "Good", "#009966", "Minimal impact"
    elif aqi <= 100:
        return "Satisfactory", "#86c72c", "Minor breathing discomfort to sensitive people"
    elif aqi <= 200:
        return "Moderately Polluted", "#ff9933", "Breathing discomfort to the people with lungs, asthma and heart diseases"
    elif aqi <= 300:
        return "Poor", "#cc0033", "Breathing discomfort to most people on prolonged exposure"
    elif aqi <= 400:
        return "Very Poor", "#660099", "Respiratory illness on prolonged exposure"
    else:
        return "Severe", "#7e0023", "Affects healthy people and seriously impacts those with existing diseases"

def add_aqi_columns(df):
    """
    Appends overall AQI and sub-indices to a dataframe containing pollutant columns.
    """
    df = df.copy()
    
    # Add individual sub-indices
    if "PM2.5" in df.columns:
        df["PM2.5_SubIndex"] = df["PM2.5"].apply(get_pm25_sub_index)
    if "PM10" in df.columns:
        df["PM10_SubIndex"] = df["PM10"].apply(get_pm10_sub_index)
    if "NO2" in df.columns:
        df["NO2_SubIndex"] = df["NO2"].apply(get_no2_sub_index)
    if "SO2" in df.columns:
        df["SO2_SubIndex"] = df["SO2"].apply(get_so2_sub_index)
    if "CO" in df.columns:
        df["CO_SubIndex"] = df["CO"].apply(get_co_sub_index)
    if "O3" in df.columns:
        df["O3_SubIndex"] = df["O3"].apply(get_o3_sub_index)
        
    # Add overall AQI
    df["AQI"] = df.apply(calculate_cpcb_aqi, axis=1)
    
    # Add category and color
    cat_details = df["AQI"].apply(get_aqi_category)
    df["AQI_Category"] = [c[0] for c in cat_details]
    df["AQI_Color"] = [c[1] for c in cat_details]
    
    return df
