import os
import sys

# Append current directory to path to handle imports properly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_simulator import run_simulation
from src.preprocessing import align_and_match
from src.ml_models import train_and_evaluate_ml
from src.dl_models import train_and_evaluate_dl
from src.hotspot_detection import analyze_fire_hcho_lag_correlation

def main():
    base_dir = "c:/Users/Anjali/OneDrive/Desktop/ISRO"
    
    print("=========================================================")
    print("   AIR QUALITY ESTIMATION & HCHO HOTSPOT DETECTION       ")
    print("                PIPELINE RUNNER                          ")
    print("=========================================================\n")
    
    # Step 1: Run Data Simulation
    print("STEP 1/5: RUNNING SPATIAL-TEMPORAL DATA SIMULATION")
    run_simulation(base_dir)
    print("\n" + "="*50 + "\n")
    
    # Step 2: Run Spatiotemporal Matching
    print("STEP 2/5: RUNNING SPATIOTEMPORAL DATA MATCHING & QUALITY CONTROL")
    align_and_match(base_dir)
    print("\n" + "="*50 + "\n")
    
    # Step 3: Train & Validate Machine Learning Models (Random Forest & XGBoost)
    print("STEP 3/5: TRAINING AND EVALUATING MACHINE LEARNING REGRESSORS")
    train_and_evaluate_ml(base_dir)
    print("\n" + "="*50 + "\n")
    
    # Step 4: Train & Validate Deep Learning CNN-LSTM Model
    print("STEP 4/5: TRAINING AND EVALUATING PYTORCH CNN-LSTM MODEL")
    train_and_evaluate_dl(base_dir, epochs=15, batch_size=32)
    print("\n" + "="*50 + "\n")
    
    # Step 5: Run Hotspot and Lag-Correlation Analysis
    print("STEP 5/5: RUNNING HCHO-FIRE LAG-CORRELATION ANALYSIS")
    analyze_fire_hcho_lag_correlation(base_dir)
    print("\n" + "="*50 + "\n")
    
    print("Pipeline Execution Completed Successfully! All models and dataset outputs are saved.")

if __name__ == "__main__":
    main()
