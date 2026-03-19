import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def nasa_scoring(y_true, y_pred):
    """Asymmetric scoring: Late predictions are penalized heavier than early ones."""
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1))

def evaluate_point_predictions(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    score = nasa_scoring(y_true, y_pred)
    print(f"\n=== {name} Point Metrics ===")
    print(f"RMSE:       {rmse:.2f}")
    print(f"MAE:        {mae:.2f}")
    print(f"NASA Score: {score:.1f}")

def winkler_score(lower, upper, y_true, alpha=0.1):
    width = upper - lower
    penalty = np.maximum(lower - y_true, 0) + np.maximum(y_true - upper, 0)
    return np.mean(width + (2 / alpha) * penalty)

def evaluate_intervals(name, lower, upper, y_true):
    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
    width = np.mean(upper - lower)
    wink = winkler_score(lower, upper, y_true, alpha=0.1)
    print(f"\n=== {name} Interval Metrics ===")
    print(f"Empirical Coverage: {coverage:.1f}%")
    print(f"Avg Width:          {width:.1f} cycles")
    print(f"Winkler Score:      {wink:.1f}")