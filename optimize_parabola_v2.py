import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# High Precision Data Points provided by User
# w = Weight of M40 in blending with M39
# Score = RMSLE
data = {
    0.0000: 0.21912744, # M39
    0.2000: 0.21901124, # M43
    0.2957: 0.21899198, # M45 (Current Best)
    1.0000: 0.21957309  # M40
}
# Note: M46 excluded (Bias corrected, not a weight blend)
# Note: M47 excluded (Recursive failure)

weights = np.array(list(data.keys()))
scores = np.array(list(data.values()))

def optimize_v2():
    print("Fitting Parabola to 4 Points...")
    coeffs = np.polyfit(weights, scores, 2)
    a, b, c = coeffs
    
    print(f"Curve: {a:.6f}w^2 + {b:.6f}w + {c:.6f}")
    
    # Derivative = 0
    w_opt = -b / (2 * a)
    pred_score = a*w_opt**2 + b*w_opt + c
    
    print(f"\nNew Optimal Weight: {w_opt:.5f}")
    print(f"Predicted Score:    {pred_score:.8f}")
    
    # Check if we are just re-finding M45 (0.2957)
    diff = abs(w_opt - 0.2957)
    print(f"Shift from M45: {diff:.5f}")
    
    # Generate M48
    print("\nGenerating submission048.csv...")
    PATH_M39 = 'submission039.csv'
    PATH_M40 = 'submission040.csv'
    OUT_PATH = 'submission048.csv'
    
    # Align!
    s39 = pd.read_csv(PATH_M39).sort_values('data_id').reset_index(drop=True)
    s40 = pd.read_csv(PATH_M40).sort_values('data_id').reset_index(drop=True)
    
    w = max(0.0, min(1.0, w_opt))
    print(f"Using weight: {w:.5f}")
    
    p39 = np.log1p(s39['construction_cost_per_m2_usd'])
    p40 = np.log1p(s40['construction_cost_per_m2_usd'])
    
    log_blend = (1 - w) * p39 + w * p40
    final_pred = np.expm1(log_blend)
    
    sub = pd.DataFrame({'data_id': s39['data_id'], 'construction_cost_per_m2_usd': final_pred})
    sub.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    optimize_v2()
