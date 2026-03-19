import numpy as np

def calibrate_cqr(xgb_models, X_train, y_train, train_units, holdout_size=20, alpha_target=0.90):
    """
    Applies Conformal Prediction by simulating NASA's random truncation on a holdout set.
    Returns the q_hat penalty to apply to test predictions.
    """
    calib_units = train_units.unique()[-holdout_size:]
    
    # Random Truncation Sampling
    np.random.seed(42)
    calib_indices = []
    for unit in calib_units:
        unit_idx = X_train[train_units == unit].index
        random_cuts = np.random.choice(unit_idx, size=min(15, len(unit_idx)), replace=False)
        calib_indices.extend(random_cuts)
        
    X_calib = X_train.loc[calib_indices]
    y_calib = y_train.loc[calib_indices]
    
    # Calculate non-conformity scores
    non_conf_xgb = np.maximum(
        xgb_models[0.05].predict(X_calib) - y_calib,
        y_calib - xgb_models[0.95].predict(X_calib)
    )
    
    q_hat = np.quantile(non_conf_xgb, alpha_target)
    return q_hat

def apply_cqr_bounds(pred_05, pred_95, q_hat):
    """Applies the calculated penalty to create strict 90% bounds."""
    lower = np.clip(pred_05 - q_hat, 1.0, None) # Engines cannot have negative RUL
    upper = pred_95 + q_hat
    return lower, upper