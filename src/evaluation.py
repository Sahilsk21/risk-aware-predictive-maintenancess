"""
Evaluation Metrics Module for Predictive Maintenance.

This module provides specialized evaluation metrics for assessing the performance 
of Remaining Useful Life (RUL) prediction models. It includes metrics for both 
deterministic point estimates (RMSE, MAE, NASA Asymmetric Score) and probabilistic 
predictive intervals (Empirical Coverage, Winkler Score).


"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Union

# Define a custom type alias for array-like inputs to keep signatures clean
ArrayLike = Union[np.ndarray, list, float, int]

def nasa_scoring(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Calculates the NASA Asymmetric Scoring function for RUL predictions.
    
    In safety-critical domains like aviation, overestimating RUL (late prediction) 
    is catastrophically worse than underestimating RUL (early prediction). This 
    function applies an exponential penalty that punishes late predictions more 
    severely than early ones.

    Args:
        y_true (ArrayLike): Ground truth Remaining Useful Life values.
        y_pred (ArrayLike): Predicted Remaining Useful Life values.

    Returns:
        float: The aggregated asymmetric penalty score. Lower is better.
    """
    # Convert inputs to numpy arrays for vectorized operations
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate the error delta (Predicted - True)
    # If d < 0: Early prediction (Safe, mild penalty)
    # If d >= 0: Late prediction (Dangerous, severe penalty)
    d = y_pred - y_true
    
    # Apply the piecewise exponential penalty
    penalty = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    
    return float(np.sum(penalty))

def evaluate_point_predictions(name: str, y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """
    Evaluates and logs standard and domain-specific metrics for point predictions.

    Args:
        name (str): Identifier for the model being evaluated (used for logging).
        y_true (ArrayLike): Ground truth RUL values.
        y_pred (ArrayLike): Predicted RUL point estimates (e.g., median or mean).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    score = nasa_scoring(y_true, y_pred)
    
    print(f"\n=== {name} Point Metrics ===")
    print(f"RMSE:       {rmse:.2f}")
    print(f"MAE:        {mae:.2f}")
    print(f"NASA Score: {score:.1f}")

def winkler_score(lower: ArrayLike, upper: ArrayLike, y_true: ArrayLike, alpha: float = 0.1) -> float:
    """
    Calculates the Winkler Score to evaluate the quality of predictive intervals.
    
    The Winkler score evaluates both the sharpness (width) and the calibration (coverage)
    of a predictive interval. It rewards narrow intervals but applies a severe penalty 
    if the ground truth falls outside the predicted bounds.

    Args:
        lower (ArrayLike): Lower bounds of the predictive intervals.
        upper (ArrayLike): Upper bounds of the predictive intervals.
        y_true (ArrayLike): Ground truth values.
        alpha (float, optional): The significance level corresponding to the interval 
                                 (e.g., 0.1 for a 90% predictive interval). Defaults to 0.1.

    Returns:
        float: The average Winkler score across all samples. Lower is better.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    y_true = np.asarray(y_true)
    
    # Base penalty is the width of the interval (we want it to be sharp/narrow)
    width = upper - lower
    
    # Additional penalty applied only if the true value falls outside the bounds
    penalty = np.maximum(lower - y_true, 0) + np.maximum(y_true - upper, 0)
    
    # Total score balances the base width against the out-of-bounds penalty
    winkler_scores = width + (2 / alpha) * penalty
    
    return float(np.mean(winkler_scores))

def evaluate_intervals(name: str, lower: ArrayLike, upper: ArrayLike, y_true: ArrayLike) -> None:
    """
    Evaluates and logs metrics for probabilistic uncertainty bounds (predictive intervals).

    Args:
        name (str): Identifier for the model/method being evaluated.
        lower (ArrayLike): Lower bounds of the predictive intervals.
        upper (ArrayLike): Upper bounds of the predictive intervals.
        y_true (ArrayLike): Ground truth RUL values.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    y_true = np.asarray(y_true)
    
    # Calculate empirical coverage: The percentage of times the true RUL fell inside the bounds
    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    
    # Calculate sharpness: The average operational window provided to decision-makers
    width = np.mean(upper - lower)
    
    # Calculate the unified interval metric (assumes a 90% interval via alpha=0.1)
    wink = winkler_score(lower, upper, y_true, alpha=0.1)
    
    print(f"\n=== {name} Interval Metrics ===")
    print(f"Empirical Coverage: {coverage:.1f}%")
    print(f"Avg Width:          {width:.1f} cycles")
    print(f"Winkler Score:      {wink:.1f}")