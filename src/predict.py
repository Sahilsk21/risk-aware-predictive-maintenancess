import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

from src.features import build_features
from src.uncertainty import apply_cqr_bounds

def predict_rul(input_data_path, model_dir="data/"):
    """
    Production Inference Pipeline: 
    Loads new raw sensor data, loads saved artifacts, applies features, and predicts RUL.
    """
    print(f"=== RUNNING INFERENCE ON: {input_data_path} ===\n")
    
    # 1. Load Pre-Trained Artifacts
    # In a production environment, train.py exports these so the inference script 
    # doesn't have to recalculate global statistics or retrain models.
    try:
        print(">> Loading models, scaler, and calibration data...")
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(model_dir, 'active_sensors.pkl'), 'rb') as f:
            active_sensors = pickle.load(f)
        with open(os.path.join(model_dir, 'q_hat.pkl'), 'rb') as f:
            q_hat = pickle.load(f)
            
        model_50 = xgb.XGBRegressor()
        model_50.load_model(os.path.join(model_dir, 'xgb_model_50.json'))
        model_05 = xgb.XGBRegressor()
        model_05.load_model(os.path.join(model_dir, 'xgb_model_05.json'))
        model_95 = xgb.XGBRegressor()
        model_95.load_model(os.path.join(model_dir, 'xgb_model_95.json'))
        
    except FileNotFoundError:
        print("❌ ERROR: Model artifacts not found.")
        print("Please ensure your train.py script saves the scaler, models, and q_hat using pickle/xgboost.load_model.")
        return

    # 2. Load Raw Live Data
    print(">> Ingesting raw sensor telemetry...")
    cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    new_data = pd.read_csv(input_data_path, sep=r'\s+', header=None, names=cols)
    
    # 3. Build Features (Using the pre-fitted scaler)
    print(">> Extracting temporal features (rolling windows & lags)...")
    features_df = build_features(new_data, scaler, active_sensors)
    
    # Isolate the current cycle (the last observed row) for each engine
    current_state = features_df.groupby('unit').tail(1).sort_values('unit').reset_index(drop=True)
    
    exclude = ['unit', 'cycle', 'RUL', 'op1', 'op2', 'op3'] + active_sensors
    feature_cols = [c for c in current_state.columns if c not in exclude]
    
    X_live = current_state[feature_cols]
    
    # 4. Model Inference
    print(">> Generating Predictions and Uncertainty Bounds...")
    pred_50 = model_50.predict(X_live)
    pred_05 = model_05.predict(X_live)
    pred_95 = model_95.predict(X_live)
    
    # Apply Conformal Penalty to guarantee ~90% coverage
    lower_bound, upper_bound = apply_cqr_bounds(pred_05, pred_95, q_hat)
    
    # 5. Format Output for the Dashboard/Decision Service
    results = pd.DataFrame({
        'Engine_Unit': current_state['unit'],
        'Predicted_RUL_Median': np.round(pred_50, 1),
        'CQR_Lower_Bound': np.round(lower_bound, 1),
        'CQR_Upper_Bound': np.round(upper_bound, 1),
        'Urgency_Interval_Width': np.round(upper_bound - lower_bound, 1)
    })
    
    print("\n=== INFERENCE RESULTS (TOP 5 ENGINES) ===")
    print(results.head().to_markdown(index=False))
    
    output_file = os.path.join(model_dir, 'live_predictions.csv')
    results.to_csv(output_file, index=False)
    print(f"\n✅ Pipeline complete. Predictions pushed to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict RUL on new sensor data")
    parser.add_argument("--input", type=str, required=True, help="Path to new raw sensor data (e.g., data/test_FD002.txt)")
    parser.add_argument("--model_dir", type=str, default="data/", help="Directory containing saved model artifacts")
    args = parser.parse_args()
    
    predict_rul(args.input, args.model_dir)