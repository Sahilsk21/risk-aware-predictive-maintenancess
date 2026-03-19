import pandas as pd
import numpy as np
import os

def load_cmapss_data(data_dir, dataset="FD001", cap_rul=125.0):
    """Loads train, test, and RUL datasets, and computes the capped target."""
    
    cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    
    train_path = os.path.join(data_dir, f'train_{dataset}.txt')
    test_path = os.path.join(data_dir, f'test_{dataset}.txt')
    rul_path = os.path.join(data_dir, f'RUL_{dataset}.txt')
    
    train = pd.read_csv(train_path, sep=r'\s+', header=None, names=cols)
    test  = pd.read_csv(test_path, sep=r'\s+', header=None, names=cols)
    rul_test = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    
    # Task B: Compute True RUL and apply piecewise cap for training stability
    train['RUL'] = train.groupby('unit')['cycle'].transform('max') - train['cycle']
    if cap_rul:
        train['RUL'] = train['RUL'].clip(upper=cap_rul)
        
    y_true_test = np.clip(rul_test['RUL'].values, a_min=None, a_max=cap_rul) if cap_rul else rul_test['RUL'].values
    
    return train, test, y_true_test