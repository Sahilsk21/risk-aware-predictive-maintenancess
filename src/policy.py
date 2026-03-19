import numpy as np
import pandas as pd
import argparse
import os

def simulate_dynamic_fleet(policy_name, pred_failure_day, true_failure_day, 
                           interval_width=None, capacity=3, lead_time=15, 
                           c_pm=5000, c_f=50000, c_w=100):
    """Simulates fleet scheduling using Greedy Dynamic Assignment."""
    num_engines = len(true_failure_day)
    serviced = np.zeros(num_engines, dtype=bool)
    failed = np.zeros(num_engines, dtype=bool)
    
    total_cost, failures, prevented, wasted_life_total = 0, 0, 0, 0
    max_day = int(max(true_failure_day)) + 15 
    
    for day in range(max_day):
        # 1. Log Failures
        for i in range(num_engines):
            if not serviced[i] and not failed[i] and day >= true_failure_day[i]:
                failed[i] = True; failures += 1; total_cost += c_f
                
        # 2. Build Daily Queue
        candidates = []
        for i in range(num_engines):
            if not serviced[i] and not failed[i]:
                days_to_pred_fail = pred_failure_day[i] - day
                if days_to_pred_fail <= lead_time:
                    urgency = days_to_pred_fail
                    if interval_width is not None:
                        urgency -= 0.25 * interval_width[i] # Risk penalty
                    candidates.append((urgency, i))
                    
        # 3. Service Top K
        if candidates:
            candidates.sort(key=lambda x: x[0])
            for urgency, i in candidates[:capacity]:
                serviced[i] = True; prevented += 1; total_cost += c_pm
                waste = true_failure_day[i] - day
                wasted_life_total += waste
                total_cost += (waste * c_w)
                
    avg_waste = wasted_life_total / max(prevented, 1)
    return {
        'Policy': policy_name,
        'Total Cost ($)': total_cost,
        'Failures Avoided': prevented,
        'Unplanned Failures': failures,
        'Avg Wasted Life': round(avg_waste, 1)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--c_pm", type=int, default=5000)
    parser.add_argument("--c_fail", type=int, default=50000)
    parser.add_argument("--data_dir", type=str, default="/home/user/risk-aware-predictive-maintenance/data/archive/CMaps/")
    args = parser.parse_args()
    
    print(f"=== RUNNING STANDALONE POLICY SIMULATION ===")
    try:
        # Load directly from the directory passed in the arguments
        xgb_pred_50 = np.load(os.path.join(args.data_dir, 'xgb_pred_50.npy'))
        xgb_lower = np.load(os.path.join(args.data_dir, 'xgb_lower.npy'))
        cqr_width = np.load(os.path.join(args.data_dir, 'cqr_width.npy'))
        y_true = np.load(os.path.join(args.data_dir, 'y_true.npy'))
        
        res_naive = simulate_dynamic_fleet('Naive ML', xgb_pred_50, y_true, capacity=args.capacity)
        res_risk = simulate_dynamic_fleet('Risk-Aware', xgb_lower, y_true, interval_width=cqr_width, capacity=args.capacity)
        
        df = pd.DataFrame([res_naive, res_risk]).set_index('Policy')
        print(df.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x).to_markdown())
        
    except FileNotFoundError:
        print(f"Simulation data not found in {args.data_dir}. Run src/train.py first to generate predictions.")