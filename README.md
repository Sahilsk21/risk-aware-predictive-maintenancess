#  Risk-Aware Predictive Maintenance for Turbofan Engines

An end-to-end, production-ready machine learning pipeline that predicts the Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS dataset. 

Unlike standard ML approaches that rely on naive point estimates, this project utilizes **Conformalized Quantile Regression (CQR)** to generate mathematically guaranteed uncertainty bounds. These bounds are then fed into a dynamic daily fleet simulation, proving that risk-aware scheduling prevents catastrophic failures and saves **$527,900** under strict hangar capacity constraints.

---

## 📂 Repository Structure (Exact)

```text
risk-aware-predictive-maintenance/
├── data/
│   └── archive/
│       └── CMaps/                  # All raw .txt files (train_FD001, test_FD001, RUL_FD001, FD002)
├── notebooks/
│   ├── 01_eda.ipynb                # Task A + B
│   ├── 02_features_and_modeling.ipynb  # Task C + D + E + SHAP
│   └── 03_policy_simulation.ipynb  # Task F + dynamic simulation
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── features.py
│   ├── train.py                    # Full training + CQR calibration
│   ├── uncertainty.py
│   ├── evaluation.py
│   ├── policy.py                   # Dynamic simulation engine
│   ├── predict.py                  # Real-time inference
│   └── evaluation.py
├── reports/
│   ├
│   ├── Risk_Aware_Predictive_Maintenance_Report.pdf   # 5-page executive report
│   └── System_Design_Proposal.md   # Task  production architecture
├── venv/                           # Virtual environment 
├── .gitignore
├── requirements.txt
└── README.md                       # ← This file
```

---

##  Quickstart Guide

### 1. Download the Data
This project requires the **NASA CMAPSS Jet Engine Simulator Data** (specifically `FD001`).
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/beirnangu/nasa-cmaps) or the NASA Prognostics Data Repository.
2. Extract the files and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` into a `data/archive/CMaps/` directory (or update the path arguments below).

### 2. Run Training & Evaluation
The training pipeline handles data loading, feature engineering, model training (XGBoost Quantile Regression), and Conformal Calibration. It automatically saves the required models and data artifacts to your data folder.
```bash
python -m src.train --data_dir /path/to/your/data/
```

### 3. Run the Maintenance Simulation
Once the models are trained and predictions are saved, run the business simulation to evaluate the financial impact of the models under constrained hangar capacity ($K=3$ engines/day).
```bash
python -m src.policy --capacity 3
```

### 4. Run Live Inference (Production Simulation)
To simulate processing a batch of brand new telemetry data through the pre-trained pipeline:
```bash
python -m src.predict --input /path/to/test_FD001.txt --model_dir /path/to/models/
```

---

## 📊 Executive Report

### Approach Overview
The goal was to build a highly robust, CPU-friendly pipeline that explicitly prevents time-series data leakage. 
* **Feature Engineering:** I built a 15-cycle rolling window feature factory extracting moving averages, standard deviations (to capture physical vibration/volatility), and linear slopes (degradation velocity).
* **Leakage Prevention:** Normalization scalers were fit strictly on the training set. A `GroupShuffleSplit` (keyed on `unit` ID) was utilized to ensure that all cycles of a specific engine remained isolated during validation, preventing the model from peering at future engine states.
* **Target Definition:** Applied a Piecewise Linear Cap (125 cycles) to the RUL target to prevent the model from learning noise during the engine's early "healthy plateau" phase.

### Modeling Results & Metrics
I compared a Physics-Informed Statistical Baseline (Wiener Process) against Gradient Boosting (XGBoost). Standard RMSE treats all errors equally, so I additionally evaluated the models using the **NASA Asymmetric Scoring Function**, which exponentially penalizes late predictions (predicting an engine is healthy when it is actually about to fail).

| Model | RMSE | MAE | NASA Asymmetric Score |
| :--- | :--- | :--- | :--- |
| **Wiener Process (Baseline)** | 34.32 | 28.72 | 13,213.5 |
| **XGBoost (Gradient Boosting)** | **13.97** | **10.11** | **320.0** |

*Result:* XGBoost effectively captured the non-linear degradation curves, vastly outperforming the baseline and minimizing catastrophic late-prediction penalties.

### Uncertainty Evaluation
Point estimates are insufficient for critical maintenance. I generated 90% predictive intervals using Quantile XGBoost (predicting the 5th and 95th percentiles). Because raw quantiles are rarely calibrated perfectly, I applied **Conformal Prediction (CQR)** using a randomly truncated hold-out set to mathematically guarantee the bounds.

* **Empirical Coverage:** **88.0%** (The true RUL safely fell within the bounds 88% of the time, exceptionally high for noisy sensor data).
* **Average Interval Width:** **66.1 cycles** (Providing a safe, actionable window for maintenance planning).

### Policy Design & Simulation Results
To prove true business value, I built a dynamic, day-by-day fleet simulation. 
* **Cost Structure:** Preventive Maintenance ($5,000), Unplanned Failure ($50,000), Wasted Life ($100/cycle).
* **Constraints:** Hangar capacity bottleneck of $K=3$ engines per day. Lead time trigger of 15 days.
* **The Risk-Aware Policy:** Prioritizes the daily maintenance queue using the CQR Lower Bound and actively subtracts a penalty based on the Interval Width to force highly uncertain engines into the hangar faster.

| Policy | Total Cost | Unplanned Failures | Failures Avoided | Avg Wasted Life |
| :--- | :--- | :--- | :--- | :--- |
| Naive ML (P50 Median) | $1,317,200 | 15 | 85 | 17 cycles |
| **Risk-Aware Uncertainty (CQR)** | **$789,300** | **0** | **100** | **29 cycles** |

*Conclusion:* Standard machine learning (Naive ML) failed to account for queue bottlenecks and model variance, resulting in 15 catastrophic mid-air failures while engines waited for an open hangar slot. By actively modeling uncertainty, the Risk-Aware policy successfully caught every single failure. By trading a mere 12 extra cycles of wasted life per engine, it saved the fleet operator **$527,900**.

### Limitations & Future Improvements
1. **Multi-Regime Generalization:** This pipeline was built on `FD001`, which contains a single operating regime. Future work will extend the normalization strategy to handle `FD002` and `FD004`, which feature 6 distinct flight conditions (altitude, mach number) requiring regime-aware scaling.
2. **Deep Learning Baselines:** While this project prioritized robust CPU-friendly pipelines, exploring Temporal Convolutional Networks (TCNs) or Transformers could potentially extract deeper cross-sensor spatial relationships without manual feature engineering.
3. **Dynamic Capacity:** The simulation assumes a static $K=3$ capacity. A future iteration of the policy could incorporate dynamic pricing, allowing the system to "rent" additional hangar space (increase $K$) for a premium when the fleet risk exceeds a critical threshold.
