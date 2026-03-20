"""
Risk-Aware Predictive Maintenance Pipeline.

This package provides the core machine learning and operations research modules 
required to predict the Remaining Useful Life (RUL) of turbofan engines. It includes 
tools for leakage-free feature engineering, Conformalized Quantile Regression (CQR), 
and dynamic fleet simulation under strict capacity constraints.


"""

# Expose core data ingestion functions
from .load_data import load_cmapss_data

# Expose core feature engineering functions
from .features import extract_active_sensors, build_features

# Expose core uncertainty quantification functions
from .uncertainty import calibrate_cqr, apply_cqr_bounds

# Expose the business simulation logic
from .policy import simulate_dynamic_fleet

# Define explicitly what is exported when a user runs `from src import *`
__all__ = [
    "load_cmapss_data",
    "extract_active_sensors",
    "build_features",
    "calibrate_cqr",
    "apply_cqr_bounds",
    "simulate_dynamic_fleet",
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Muttalip Sk"