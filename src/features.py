"""
Feature Engineering Module for Turbofan Engine Degradation.

This module acts as a "Feature Factory" for time-series telemetry data. 
It processes raw sensor arrays into machine-learning-ready features by applying 
leakage-free normalization, physical signal alignment, and rolling temporal windows 
to capture the velocity and volatility of engine degradation.


"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Union

def fast_rolling_slope(y: Union[np.ndarray, pd.Series, list]) -> float:
    """
    Vectorized calculation of the linear regression slope over a rolling window.
    
    This calculates the physical 'velocity' of degradation. Instead of using 
    slow loops or heavy scipy functions, it computes the ordinary least squares (OLS) 
    slope mathematically: sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2).

    Args:
        y (Union[np.ndarray, pd.Series, list]): An array of sensor values representing 
                                                the current rolling window.

    Returns:
        float: The linear slope of the values over time. Returns 0.0 if the window 
               is too small to calculate a trend.
    """
    if len(y) < 2: 
        return 0.0
    
    # x represents the time steps within the window [0, 1, 2, ..., w-1]
    x = np.arange(len(y))
    
    # Calculate the denominator (variance of x). If 0, prevent division by zero.
    denom = np.sum((x - x.mean()) ** 2)
    if denom == 0:
        return 0.0
        
    # Return the OLS slope coefficient
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)

def extract_active_sensors(train_df: pd.DataFrame, threshold: float = 1e-4) -> List[str]:
    """
    Identifies and extracts sensors that exhibit meaningful variance.
    
    Many sensors in static operating conditions provide flat-line readings. 
    This function drops zero-variance sensors to reduce dimensionality and 
    prevent multi-collinearity issues downstream.

    Args:
        train_df (pd.DataFrame): The raw training telemetry dataframe.
        threshold (float, optional): The minimum standard deviation required to 
                                     keep the sensor. Defaults to 1e-4.

    Returns:
        List[str]: A list of sensor column names that pass the variance threshold.
    """
    sensor_cols = [f's{i}' for i in range(1, 22)]
    stds = train_df[sensor_cols].std()
    
    # Filter and return only the sensors with a standard deviation above the threshold
    active_sensors = stds[stds > threshold].index.tolist()
    return active_sensors

def build_features(
    df: pd.DataFrame, 
    scaler: MinMaxScaler, 
    active_sensors: List[str], 
    train_ref: Optional[pd.DataFrame] = None, 
    window: int = 15
) -> pd.DataFrame:
    """
    Master pipeline for feature extraction, alignment, and temporal windowing.
    
    This function handles the end-to-end transformation of raw cycle data into 
    predictive features. It strictly enforces leakage prevention by relying on 
    pre-fitted scalers and reference correlations.

    Args:
        df (pd.DataFrame): The raw telemetry dataframe (Train, Test, or Live Inference).
        scaler (MinMaxScaler): A pre-fitted scaler calibrated ONLY on training data.
        active_sensors (List[str]): List of sensors to process (output of `extract_active_sensors`).
        train_ref (Optional[pd.DataFrame], optional): The training dataframe used to calculate 
                                                      correlation directions. Defaults to None.
        window (int, optional): The size of the rolling window for temporal stats. Defaults to 15.

    Returns:
        pd.DataFrame: A highly enriched dataframe containing smoothed means, physical 
                      volatility (std), degradation velocity (slope), and health indices.
    """
    # Work on a copy to prevent SettingWithCopyWarnings
    df_feat = df.copy()
    
    # -------------------------------------------------------------------------
    # 1. Leakage-Free Normalization
    # -------------------------------------------------------------------------
    df_feat[active_sensors] = scaler.transform(df_feat[active_sensors])
    
    # -------------------------------------------------------------------------
    # 2. Signal Alignment
    # -------------------------------------------------------------------------
    # Turbofan sensors degrade in different directions (some heat up, some lose pressure).
    # We align all sensors to a unified degradation vector (trending upwards) by 
    # checking their correlation to 'cycle' in the reference training data.
    if train_ref is not None:
        for col in active_sensors:
            if train_ref['cycle'].corr(train_ref[col]) < 0:
                # Invert the normalized signal (since values are scaled 0 to 1)
                df_feat[col] = 1.0 - df_feat[col]
                
    # -------------------------------------------------------------------------
    # 3. Macro Health Indicators
    # -------------------------------------------------------------------------
    # Global Health Index: The mean of all aligned sensors
    df_feat['HI'] = df_feat[active_sensors].mean(axis=1)
    
    # Early Deviation: Compares the current cycle to the engine's "healthy plateau" (first 30 cycles)
    for col in active_sensors:
        df_feat[f'{col}_early_dev'] = df_feat[col] - df_feat.groupby('unit')[col].transform(lambda x: x.head(30).mean())
        
    # -------------------------------------------------------------------------
    # 4. Temporal Features (Lags & Rolling Windows)
    # -------------------------------------------------------------------------
    # We extract temporal context to capture the dynamic physical state of the engine
    for col in active_sensors:
        grp = df_feat.groupby('unit')[col]
        
        # Historical Lags (Lookback)
        for lag in [1, 5]:
            df_feat[f'{col}_lag{lag}'] = grp.shift(lag).bfill()
        
        # Moving Average (Smoothing noise)
        df_feat[f'{col}_mean'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        
        # Rolling Std (Physical vibration/volatility indicator)
        df_feat[f'{col}_std']  = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        
        # Rolling Max (Peak stress indicator)
        df_feat[f'{col}_max']  = grp.transform(lambda x: x.rolling(window, min_periods=1).max())
        
        # Rolling Slope (Velocity/Acceleration of degradation)
        df_feat[f'{col}_slope'] = grp.transform(lambda x: x.rolling(window, min_periods=2).apply(fast_rolling_slope, raw=True).fillna(0))
        
    return df_feat