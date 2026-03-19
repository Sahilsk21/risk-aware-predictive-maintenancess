import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

from src.load_data import load_cmapss_data
from src.features import extract_active_sensors, build_features
from src.uncertainty import calibrate_cqr, apply_cqr_bounds
from src.evaluation import evaluate_point_predictions, evaluate_intervals

def main(args):
    print(f"=== INITIATING TRAINING PIPELINE ({args.dataset}) ===")
    
    # 1. Load Data
    train, test, y_true = load_cmapss_data(args.data_dir, dataset=args.dataset)
    
    # 2. Identify Sensors & Fit Scaler
    active_sensors = extract_active_sensors(train)
    scaler = MinMaxScaler()
    scaler.fit(train[active_sensors])
    
    # 3. Build Features
    print(">> Extracting time-series features...")
    train_features = build_features(train, scaler, active_sensors, train_ref=train)
    test_features = build_features(test, scaler, active_sensors, train_ref=train)
    
    exclude = ['unit', 'cycle', 'RUL', 'op1', 'op2', 'op3'] + active_sensors
    feature_cols = [c for c in train_features.columns if c not in exclude]
    
    X_train, y_train = train_features[feature_cols], train_features['RUL']
    
    test_last = test_features.groupby('unit').tail(1).sort_values('unit').reset_index(drop=True)
    X_test_last = test_last[feature_cols]
    
    # 4. Train Quantile Models & Calibrate
    print(">> Training Quantile XGBoost Regressors...")
    calib_units = train_features['unit'].unique()[-20:]
    fit_mask = ~train_features['unit'].isin(calib_units)
    
    xgb_models = {}
    for alpha in [0.05, 0.50, 0.95]:
        model = xgb.XGBRegressor(objective='reg:quantileerror', n_estimators=800, max_depth=7, 
                                 learning_rate=0.04, subsample=0.85, random_state=42, n_jobs=-1, quantile_alpha=alpha)
        model.fit(X_train[fit_mask], y_train[fit_mask])
        xgb_models[alpha] = model
        
    print(">> Applying Conformal Prediction (CQR)...")
    q_hat = calibrate_cqr(xgb_models, X_train, y_train, train_features['unit'])
    
    pred_05 = xgb_models[0.05].predict(X_test_last)
    pred_50 = xgb_models[0.50].predict(X_test_last)
    pred_95 = xgb_models[0.95].predict(X_test_last)
    
    lower_bound, upper_bound = apply_cqr_bounds(pred_05, pred_95, q_hat)
    cqr_width = upper_bound - lower_bound
    
    # 5. Evaluate
    evaluate_point_predictions("XGBoost Median", y_true, pred_50)
    evaluate_intervals("XGBoost CQR (90%)", lower_bound, upper_bound, y_true)
    
    # 6. Save Artifacts for Policy Simulation & Production Inference
    print(f"\n>> Saving predictions and model artifacts to {args.data_dir}...")
    
    # A. Save numpy arrays for the local policy simulation (policy.py)
    np.save(os.path.join(args.data_dir, 'xgb_pred_50.npy'), pred_50)
    np.save(os.path.join(args.data_dir, 'xgb_lower.npy'), lower_bound)
    np.save(os.path.join(args.data_dir, 'cqr_width.npy'), cqr_width)
    np.save(os.path.join(args.data_dir, 'y_true.npy'), y_true)
    
    # B. Save the actual models and preprocessing states for inference (predict.py)
    with open(os.path.join(args.data_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.data_dir, 'active_sensors.pkl'), 'wb') as f:
        pickle.dump(active_sensors, f)
    with open(os.path.join(args.data_dir, 'q_hat.pkl'), 'wb') as f:
        pickle.dump(q_hat, f)
        
    xgb_models[0.50].save_model(os.path.join(args.data_dir, 'xgb_model_50.json'))
    xgb_models[0.05].save_model(os.path.join(args.data_dir, 'xgb_model_05.json'))
    xgb_models[0.95].save_model(os.path.join(args.data_dir, 'xgb_model_95.json'))
    
    print(" Pipeline complete! Models and metrics are ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--data_dir", type=str, default="/home/user/risk-aware-predictive-maintenance/data/archive/CMaps/")
    args = parser.parse_args()
    main(args)