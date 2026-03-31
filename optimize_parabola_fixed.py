import numpy as np
import pandas as pd

# Data Points (Weight of M40)
weights = np.array([0.0, 0.2, 1.0])
scores  = np.array([0.2191, 0.21901, 0.21957])

def optimize():
    coeffs = np.polyfit(weights, scores, 2)
    a, b, c = coeffs
    w_opt = -b / (2 * a)
    pred_score = a*w_opt**2 + b*w_opt + c
    
    print(f"Optimal Weight for M40: {w_opt:.4f}")
    
    print("\nGenerating submission045.csv (FIXED ALIGNMENT)...")
    PATH_M39 = 'submission039.csv'
    PATH_M40 = 'submission040.csv'
    OUT_PATH = 'submission045.csv'
    
    # CRITICAL FIX: reset_index(drop=True)
    s39 = pd.read_csv(PATH_M39).sort_values('data_id').reset_index(drop=True)
    s40 = pd.read_csv(PATH_M40).sort_values('data_id').reset_index(drop=True)
    
    # Check alignment just to be paranoid
    if not s39['data_id'].equals(s40['data_id']):
        raise ValueError("Data IDs do not match after sort!")
        
    p39 = np.log1p(s39['construction_cost_per_m2_usd'])
    p40 = np.log1p(s40['construction_cost_per_m2_usd'])
    
    w = max(0.0, min(1.0, w_opt))
    print(f"Using weight: {w:.4f}")
    
    log_blend = (1 - w) * p39 + w * p40
    final_pred = np.expm1(log_blend)
    
    sub = pd.DataFrame({'data_id': s39['data_id'], 'construction_cost_per_m2_usd': final_pred})
    sub.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    optimize()
