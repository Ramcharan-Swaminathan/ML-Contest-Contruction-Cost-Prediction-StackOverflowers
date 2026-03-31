import pandas as pd
import numpy as np
import argparse

def ensemble(args):
    # Load submissions
    sub09 = pd.read_csv(args.sub09)
    sub10 = pd.read_csv(args.sub10)
    
    # Ensure ID alignment
    # Assuming both sorted by original index or data_id, but prompt doesn't strictly guarantee safely.
    # However, all predict scripts iterate through the evaluation csv in the same order.
    
    p09 = sub09['construction_cost_per_m2_usd'].values
    p10 = sub10['construction_cost_per_m2_usd'].values
    
    # Log Transform for averaging (RMSLE optimization)
    # Averaging in log-space is usually geometric mean in linear space, which is better for skewed targets.
    log_p09 = np.log1p(p09)
    log_p10 = np.log1p(p10)
    
    # Weighted Average
    # Method 09 (0.2220) is much stronger than Method 10 (0.4029)
    # We trust Method 09 significantly more.
    w09 = 0.95
    w10 = 0.05
    
    avg_log_pred = (w09 * log_p09) + (w10 * log_p10)
    final_pred = np.expm1(avg_log_pred)
    
    # Save
    submission = pd.DataFrame()
    if 'data_id' in sub09.columns:
        submission['data_id'] = sub09['data_id']
    submission['construction_cost_per_m2_usd'] = final_pred
    
    submission.to_csv(args.output, index=False)
    print(f"Ensemble saved to {args.output}")
    print(f"Weights: Method 09 ({w09}) | Method 10 ({w10})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub09', type=str, default='submission009.csv', help='Path to Method 09 submission')
    parser.add_argument('--sub10', type=str, default='submission010.csv', help='Path to Method 10 submission')
    parser.add_argument('--output', type=str, default='submission_final.csv')
    
    args = parser.parse_args()
    ensemble(args)
