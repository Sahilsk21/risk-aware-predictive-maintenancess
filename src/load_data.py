"""
Data Ingestion and Target Engineering Module.

This module handles the loading of the NASA CMAPSS turbofan engine dataset.
It applies standard column naming conventions and computes the target variable:
Remaining Useful Life (RUL). Crucially, it applies a Piecewise Linear Cap to 
the RUL to stabilize early-stage training.

"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional

def load_cmapss_data(
    data_dir: str, 
    dataset: str = "FD001", 
    cap_rul: Optional[float] = 125.0
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Loads train, test, and ground-truth RUL datasets and computes the target.
    
    The function automatically applies column names representing the operational 
    settings and 21 sensor readings. For the training set, it calculates the 
    true RUL per cycle and optionally applies a piecewise linear cap.

    Args:
        data_dir (str): The directory path containing the CMAPSS text files.
        dataset (str, optional): The specific sub-dataset to load (e.g., 'FD001', 
                                 'FD002'). Defaults to "FD001".
        cap_rul (Optional[float], optional): The maximum threshold to cap the RUL. 
                                             If None, no capping is applied. 
                                             Defaults to 125.0.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: 
            - train (pd.DataFrame): Training data with computed 'RUL' column.
            - test (pd.DataFrame): Test data (without RUL column).
            - y_true_test (np.ndarray): Ground truth RUL array for the test set.
    """
    # CMAPSS standard column naming convention
    # op1-3: Operational settings (altitude, mach number, throttle)
    # s1-21: Sensor measurements (temperature, pressure, fan speed, etc.)
    cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    
    # Construct file paths safely across operating systems
    train_path = os.path.join(data_dir, f'train_{dataset}.txt')
    test_path = os.path.join(data_dir, f'test_{dataset}.txt')
    rul_path = os.path.join(data_dir, f'RUL_{dataset}.txt')
    
    # Load the raw space-separated text files
    train = pd.read_csv(train_path, sep=r'\s+', header=None, names=cols)
    test  = pd.read_csv(test_path, sep=r'\s+', header=None, names=cols)
    rul_test = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    
    # -------------------------------------------------------------------------
    # Target Engineering: Compute True RUL & Piecewise Linear Cap 
    # -------------------------------------------------------------------------
    
    # 1. Calculate the absolute RUL: (Max Cycle for that unit) - (Current Cycle)
    train['RUL'] = train.groupby('unit')['cycle'].transform('max') - train['cycle']
    
    # 2. Apply Piecewise Linear Cap
    # WHY? Engines operate in a stable "healthy plateau" for most of their early life.
    # Forcing the model to predict a perfectly linear decline from Day 1 introduces 
    # severe noise. Capping forces the model to treat all early stages identically 
    # and focus its gradient updates entirely on the late-stage degradation curve.
    if cap_rul:
        train['RUL'] = train['RUL'].clip(upper=cap_rul)
        
        # Apply the exact same cap to the test ground-truth to ensure fair evaluation metrics
        y_true_test = np.clip(rul_test['RUL'].values, a_min=None, a_max=cap_rul)
    else:
        y_true_test = rul_test['RUL'].values
    
    return train, test, y_true_test