import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fast_rolling_slope(y):
    """Vectorized calculation of linear slope over a rolling window."""
    if len(y) < 2: return 0.0
    x = np.arange(len(y))
    denom = np.sum((x - x.mean())**2)
    return np.sum((x - x.mean()) * (y - y.mean())) / denom if denom != 0 else 0.0

def extract_active_sensors(train_df, threshold=1e-4):
    """Drops sensors with zero variance."""
    sensor_cols = [f's{i}' for i in range(1, 22)]
    stds = train_df[sensor_cols].std()
    return stds[stds > threshold].index.tolist()

def build_features(df, scaler, active_sensors, train_ref=None, window=15):
    """Applies normalization, signal alignment, health index, and rolling stats."""
    df_feat = df.copy()
    
    # 1. Leakage-Free Normalization
    df_feat[active_sensors] = scaler.transform(df_feat[active_sensors])
    
    # 2. Signal Alignment (Use training correlations to flip negative trends)
    if train_ref is not None:
        for col in active_sensors:
            if train_ref['cycle'].corr(train_ref[col]) < 0:
                df_feat[col] = 1.0 - df_feat[col]
                
    # 3. Health Indicators
    df_feat['HI'] = df_feat[active_sensors].mean(axis=1)
    for col in active_sensors:
        df_feat[f'{col}_early_dev'] = df_feat[col] - df_feat.groupby('unit')[col].transform(lambda x: x.head(30).mean())
        
    # 4. Temporal Features (Lags & Rolling Windows)
    for col in active_sensors:
        grp = df_feat.groupby('unit')[col]
        for lag in [1, 5]:
            df_feat[f'{col}_lag{lag}'] = grp.shift(lag).bfill()
        
        df_feat[f'{col}_mean'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df_feat[f'{col}_std']  = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df_feat[f'{col}_max']  = grp.transform(lambda x: x.rolling(window, min_periods=1).max())
        df_feat[f'{col}_slope'] = grp.transform(lambda x: x.rolling(window, min_periods=2).apply(fast_rolling_slope, raw=True).fillna(0))
        
    return df_feat