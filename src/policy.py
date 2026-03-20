"""
Decision Intelligence and Fleet Simulation Module.

This module bridges the gap between machine learning predictions and real-world 
operations. It simulates a dynamic, cycle-by-cycle maintenance hangar under strict 
capacity constraints to calculate the true financial impact of different scheduling policies.


"""

import numpy as np
import pandas as pd
import argparse
import os
from typing import Dict, Any, Optional, Union

# Define a custom type alias for array-like inputs
ArrayLike = Union[np.ndarray, list, pd.Series]

def simulate_dynamic_fleet(
    policy_name: str, 
    pred_failure_day: ArrayLike, 
    true_failure_day: ArrayLike, 
    interval_width: Optional[ArrayLike] = None, 
    capacity: int = 3, 
    lead_time: int = 15, 
    c_pm: float = 5000.0, 
    c_f: float = 50000.0, 
    c_w: float = 100.0
) -> Dict[str, Any]:
    """
    Simulates a daily fleet scheduling environment using Greedy Dynamic Assignment.
    
    This function replays the lifespan of a fleet of engines. Every "day" (cycle), it 
    evaluates which engines are approaching failure, builds a maintenance queue, and 
    services the top `K` engines based on the available hangar capacity.

    Args:
        policy_name (str): The identifier for the scheduling policy (e.g., 'Risk-Aware').
        pred_failure_day (ArrayLike): The model's predicted failure cycles for the fleet.
        true_failure_day (ArrayLike): The actual ground-truth failure cycles (for evaluation).
        interval_width (Optional[ArrayLike], optional): The width of the uncertainty interval. 
                                                        Used to penalize highly uncertain predictions.
        capacity (int, optional): Max engines the hangar can service per day. Defaults to 3.
        lead_time (int, optional): Days before predicted failure an engine enters the queue. Defaults to 15.
        c_pm (float, optional): Cost of planned Preventive Maintenance. Defaults to 5000.
        c_f (float, optional): Cost of catastrophic Unplanned Failure. Defaults to 50000.
        c_w (float, optional): Cost penalty per cycle of Wasted Life. Defaults to 100.

    Returns:
        Dict[str, Any]: A dictionary containing the financial and operational KPIs.
    """
    pred_failure_day = np.asarray(pred_failure_day)
    true_failure_day = np.asarray(true_failure_day)
    if interval_width is not None:
        interval_width = np.asarray(interval_width)
        
    num_engines = len(true_failure_day)
    serviced = np.zeros(num_engines, dtype=bool)
    failed = np.zeros(num_engines, dtype=bool)
    
    total_cost, failures, prevented, wasted_life_total = 0, 0, 0, 0
    
    # Run the simulation until the longest-living engine fails + safety buffer
    max_day = int(np.max(true_failure_day)) + 15 
    
    for day in range(max_day):
        # ---------------------------------------------------------------------
        # STEP 1: Log Unplanned Failures
        # ---------------------------------------------------------------------
        for i in range(num_engines):
            if not serviced[i] and not failed[i] and day >= true_failure_day[i]:
                failed[i] = True
                failures += 1
                total_cost += c_f  # Massive catastrophic failure penalty
                
        # ---------------------------------------------------------------------
        # STEP 2: Build the Daily Queue
        # ---------------------------------------------------------------------
        candidates = []
        for i in range(num_engines):
            if not serviced[i] and not failed[i]:
                days_to_pred_fail = pred_failure_day[i] - day
                
                # If engine is within the lead time threshold, it enters the queue
                if days_to_pred_fail <= lead_time:
                    # Base Urgency: Lower score = higher priority
                    urgency = days_to_pred_fail
                    
                    # Risk Penalty: Subtract a fraction of the uncertainty width.
                    # This mathematically forces highly volatile/uncertain engines to the 
                    # front of the line, avoiding capacity bottleneck disasters.
                    if interval_width is not None:
                        urgency -= 0.25 * interval_width[i] 
                        
                    candidates.append((urgency, i))
                    
        # ---------------------------------------------------------------------
        # STEP 3: Greedy Dynamic Assignment (Service Top K)
        # ---------------------------------------------------------------------
        if candidates:
            # Sort queue by urgency ascending (lowest urgency score is serviced first)
            candidates.sort(key=lambda x: x[0])
            
            for urgency, i in candidates[:capacity]:
                serviced[i] = True
                prevented += 1
                total_cost += c_pm
                
                # Calculate the "Wasted Life" penalty (servicing early throws away uptime)
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
    parser = argparse.ArgumentParser(description="Run the Maintenance Fleet Simulation.")
    parser.add_argument("--capacity", type=int, default=3, help="Max engines serviced per day.")
    parser.add_argument("--c_pm", type=int, default=5000, help="Preventive Maintenance Cost.")
    parser.add_argument("--c_fail", type=int, default=50000, help="Unplanned Failure Cost.")
    parser.add_argument("--data_dir", type=str, default="/home/user/risk-aware-predictive-maintenancess/data/archive/CMaps/", help="Directory containing the saved prediction arrays.")
    args = parser.parse_args()
    
    print(f"=== RUNNING STANDALONE POLICY SIMULATION ===")
    try:
        # Load directly from the directory passed in the arguments
        xgb_pred_50 = np.load(os.path.join(args.data_dir, 'xgb_pred_50.npy'))
        xgb_lower = np.load(os.path.join(args.data_dir, 'xgb_lower.npy'))
        cqr_width = np.load(os.path.join(args.data_dir, 'cqr_width.npy'))
        y_true = np.load(os.path.join(args.data_dir, 'y_true.npy'))
        
        # Simulate Naive baseline
        res_naive = simulate_dynamic_fleet('Naive ML', xgb_pred_50, y_true, capacity=args.capacity)
        # Simulate Risk-Aware strategy
        res_risk = simulate_dynamic_fleet('Risk-Aware', xgb_lower, y_true, interval_width=cqr_width, capacity=args.capacity)
        
        # Compile and format results
        df = pd.DataFrame([res_naive, res_risk]).set_index('Policy')
        
        # Note: df.map() is used here instead of applymap() for Pandas 3.0 future-proofing
        print(df.map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x).to_markdown())
        
    except FileNotFoundError:
        print(f" Simulation data not found in {args.data_dir}. Run src/train.py first to generate predictions.")