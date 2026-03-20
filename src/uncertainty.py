"""
Uncertainty Quantification and Conformal Prediction Module.

This module applies Conformalized Quantile Regression (CQR) to generate mathematically 
guaranteed predictive bounds. Raw quantile regressions are often poorly calibrated on 
unseen data. By calculating non-conformity scores on a simulated holdout set, this 
module derives an empirical penalty (q_hat) that expands or contracts the bounds to 
guarantee a specific operational coverage rate (e.g., 90%).


"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

# Define custom type aliases for cleaner function signatures
ArrayLike = np.ndarray

def calibrate_cqr(
    xgb_models: Dict[float, Any], 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    train_units: pd.Series, 
    holdout_size: int = 20, 
    alpha_target: float = 0.90
) -> float:
    """
    Calibrates Conformal Prediction bounds using a simulated random-truncation holdout.
    
    The CMAPSS test set consists of engine trajectories truncated at random operational 
    cycles. To satisfy the exchangeability requirement of Conformal Prediction, the 
    calibration set must precisely mimic this testing distribution. This function 
    isolates a holdout set, randomly truncates their histories, and calculates the 
    required empirical penalty.

    Args:
        xgb_models (Dict[float, Any]): Dictionary of trained XGBoost quantile models 
                                       keyed by their alpha (e.g., 0.05, 0.50, 0.95).
        X_train (pd.DataFrame): The training feature matrix.
        y_train (pd.Series): The ground truth RUL target values.
        train_units (pd.Series): The engine unit identifiers corresponding to X_train.
        holdout_size (int, optional): Number of engines to isolate for calibration. 
                                      Defaults to 20.
        alpha_target (float, optional): The target coverage rate (e.g., 0.90 for 90%). 
                                        Defaults to 0.90.

    Returns:
        float: The calculated Conformal penalty (q_hat) to be applied to raw quantiles.
    """
    # Isolate the final 'N' engines for calibration. 
    # CRITICAL: These engines MUST NOT have been seen by the models during training.
    calib_units = train_units.unique()[-holdout_size:]
    
    # -------------------------------------------------------------------------
    # 1. Random Truncation Sampling (Distribution Matching)
    # -------------------------------------------------------------------------
    np.random.seed(42)
    calib_indices = []
    
    for unit in calib_units:
        unit_idx = X_train[train_units == unit].index
        # Randomly sample up to 15 cycles from this engine's history to mimic 
        # the operational reality of the test set.
        random_cuts = np.random.choice(unit_idx, size=min(15, len(unit_idx)), replace=False)
        calib_indices.extend(random_cuts)
        
    X_calib = X_train.loc[calib_indices]
    y_calib = y_train.loc[calib_indices]
    
    # -------------------------------------------------------------------------
    # 2. Non-Conformity Score Calculation
    # -------------------------------------------------------------------------
    # The non-conformity score measures the maximum error by which the true RUL 
    # fell OUTSIDE the raw predicted quantile bounds.
    # Formula: E_i = max(Lower_Pred - True_y, True_y - Upper_Pred)
    non_conf_xgb = np.maximum(
        xgb_models[0.05].predict(X_calib) - y_calib,
        y_calib - xgb_models[0.95].predict(X_calib)
    )
    
    # -------------------------------------------------------------------------
    # 3. Empirical Quantile (q_hat)
    # -------------------------------------------------------------------------
    # We find the specific penalty value that covers the target percentage (e.g., 90%) 
    # of the calibration errors.
    q_hat = float(np.quantile(non_conf_xgb, alpha_target))
    
    return q_hat

def apply_cqr_bounds(
    pred_05: ArrayLike, 
    pred_95: ArrayLike, 
    q_hat: float
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Applies the Conformal Prediction penalty to generate strict operational bounds.
    
    This applies the calculated `q_hat` uniformly to the raw quantiles. It also 
    enforces physical operational logic (an engine cannot mathematically have a 
    negative Remaining Useful Life).

    Args:
        pred_05 (ArrayLike): The raw lower-bound predictions (5th percentile).
        pred_95 (ArrayLike): The raw upper-bound predictions (95th percentile).
        q_hat (float): The conformal penalty calculated by `calibrate_cqr`.

    Returns:
        Tuple[ArrayLike, ArrayLike]: 
            - lower (ArrayLike): The calibrated, physically constrained lower bounds.
            - upper (ArrayLike): The calibrated upper bounds.
    """
    pred_05 = np.asarray(pred_05)
    pred_95 = np.asarray(pred_95)
    
    # Apply penalty and enforce physical constraints (RUL must be >= 1.0)
    # If a model predicts an engine will fail in -5 days, we clip it to 1 day.
    lower = np.clip(pred_05 - q_hat, 1.0, None) 
    
    # Apply penalty to the upper bound
    upper = pred_95 + q_hat
    
    return lower, upper