"""
Master Training and Calibration Pipeline.

This orchestrator script executes the end-to-end model training process. It loads 
the CMAPSS telemetry, applies stateless feature engineering, trains three separate 
Quantile XGBoost regressors, calibrates the Conformal Prediction penalty bounds, 
and serializes all artifacts for production inference and policy simulation.


"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import Any

# Custom module imports
from src.load_data import load_cmapss_data
from src.features import extract_active_sensors, build_features
from src.uncertainty import calibrate_cqr, apply_cqr_bounds
from src.evaluation import evaluate_point_predictions, evaluate_intervals

def main(args: argparse.Namespace) -> None:
    """
    Executes the batch training, conformal calibration, and artifact serialization.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing the 
                                   dataset identifier and data directory path.
    """
    print(f"=== INITIATING TRAINING PIPELINE ({args.dataset}) ===\n")
    
    # -------------------------------------------------------------------------
    # 1. Data Ingestion
    # -------------------------------------------------------------------------
    print(">> Loading CMAPSS telemetry data...")
    train, test, y_true = load_cmapss_data(args.data_dir, dataset=args.dataset)
    
    # -------------------------------------------------------------------------
    # 2. Sensor Identification & Leakage-Free Scaling
    # -------------------------------------------------------------------------
    # Identify sensors with actual variance, ignoring flat-line sensors.
    active_sensors = extract_active_sensors(train)
    
    # CRITICAL: Fit the scaler strictly on the training distribution. 
    # Fitting on test data causes look-ahead bias (data leakage).
    scaler = MinMaxScaler()
    scaler.fit(train[active_sensors])
    
    # -------------------------------------------------------------------------
    # 3. Feature Engineering & State Isolation
    # -------------------------------------------------------------------------
    print(">> Extracting time-series features (Rolling windows & Physical velocities)...")
    train_features = build_features(train, scaler, active_sensors, train_ref=train)
    test_features = build_features(test, scaler, active_sensors, train_ref=train)
    
    # Define features to explicitly exclude from model training
    exclude = ['unit', 'cycle', 'RUL', 'op1', 'op2', 'op3'] + active_sensors
    feature_cols = [c for c in train_features.columns if c not in exclude]
    
    X_train, y_train = train_features[feature_cols], train_features['RUL']
    
    # Isolate the final recorded cycle for each engine in the test set.
    # In the real world, we only predict RUL from the current (latest) known state.
    test_last = test_features.groupby('unit').tail(1).sort_values('unit').reset_index(drop=True)
    X_test_last = test_last[feature_cols]
    
    # -------------------------------------------------------------------------
    # 4. Quantile Model Training & Conformal Calibration
    # -------------------------------------------------------------------------
    print(">> Training Quantile XGBoost Regressors...")
    
    # Hold out the last 20 engines strictly for Conformal Prediction calibration.
    # The models will NOT train on these engines, preventing overconfident bounds.
    calib_units = train_features['unit'].unique()[-20:]
    fit_mask = ~train_features['unit'].isin(calib_units)
    
    xgb_models = {}
    # Train three separate models to form our base predictive interval
    for alpha in [0.05, 0.50, 0.95]:
        model = xgb.XGBRegressor(
            objective='reg:quantileerror', 
            n_estimators=800, 
            max_depth=7, 
            learning_rate=0.04, 
            subsample=0.85, 
            random_state=42, 
            n_jobs=-1, 
            quantile_alpha=alpha
        )
        model.fit(X_train[fit_mask], y_train[fit_mask])
        xgb_models[alpha] = model
        
    print(">> Applying Conformal Prediction (CQR)...")
    # Calculate the mathematical penalty required to guarantee ~90% empirical coverage
    q_hat = calibrate_cqr(xgb_models, X_train, y_train, train_features['unit'])
    
    # Generate predictions on the unseen test set
    pred_05 = xgb_models[0.05].predict(X_test_last)
    pred_50 = xgb_models[0.50].predict(X_test_last)
    pred_95 = xgb_models[0.95].predict(X_test_last)
    
    # Expand the raw quantiles using the Conformal penalty
    lower_bound, upper_bound = apply_cqr_bounds(pred_05, pred_95, q_hat)
    cqr_width = upper_bound - lower_bound
    
    # -------------------------------------------------------------------------
    # 5. Performance Evaluation
    # -------------------------------------------------------------------------
    evaluate_point_predictions("XGBoost Median", y_true, pred_50)
    evaluate_intervals("XGBoost CQR (90%)", lower_bound, upper_bound, y_true)
    
    # -------------------------------------------------------------------------
    # 6. Artifact Serialization (MLOps Export)
    # -------------------------------------------------------------------------
    print(f"\n>> Saving predictions and model artifacts to {args.data_dir}...")
    
    # A. Export numpy arrays for local policy simulation (Task F)
    np.save(os.path.join(args.data_dir, 'xgb_pred_50.npy'), pred_50)
    np.save(os.path.join(args.data_dir, 'xgb_lower.npy'), lower_bound)
    np.save(os.path.join(args.data_dir, 'cqr_width.npy'), cqr_width)
    np.save(os.path.join(args.data_dir, 'y_true.npy'), y_true)
    
    # B. Export frozen state artifacts for live production inference (Task G)
    with open(os.path.join(args.data_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.data_dir, 'active_sensors.pkl'), 'wb') as f:
        pickle.dump(active_sensors, f)
    with open(os.path.join(args.data_dir, 'q_hat.pkl'), 'wb') as f:
        pickle.dump(q_hat, f)
        
    xgb_models[0.50].save_model(os.path.join(args.data_dir, 'xgb_model_50.json'))
    xgb_models[0.05].save_model(os.path.join(args.data_dir, 'xgb_model_05.json'))
    xgb_models[0.95].save_model(os.path.join(args.data_dir, 'xgb_model_95.json'))
    
    print(" Pipeline complete! Models and metrics are ready for production.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Training Pipeline for Turbofan RUL Prediction.")
    parser.add_argument("--dataset", type=str, default="FD001", help="CMAPSS dataset identifier.")
    parser.add_argument("--data_dir", type=str, default="/home/user/risk-aware-predictive-maintenancess/data/archive/CMaps/", help="Absolute path to the data directory.")
    args = parser.parse_args()
    
    main(args)