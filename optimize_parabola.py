import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Points (Weight of M40)
weights = np.array([0.0, 0.2, 1.0])
scores  = np.array([0.2191, 0.21901, 0.21957])

def optimize():
    # Fit parabola: score = a*w^2 + b*w + c
    coeffs = np.polyfit(weights, scores, 2)
    a, b, c = coeffs
    
    print(f"Fitted Curve: RMSE(w) = {a:.5f}w^2 + {b:.5f}w + {c:.5f}")
    
    # Derivative = 2aw + b = 0  => w = -b / (2a)
    w_opt = -b / (2 * a)
    
    pred_score = a*w_opt**2 + b*w_opt + c
    
    print(f"Optimal Weight for M40: {w_opt:.4f}")
    print(f"Predicted Score: {pred_score:.5f}")
    
    # Generate Submission
    print("\nGenerating submission044.csv...")
    PATH_M39 = 'submission039.csv'
    PATH_M40 = 'submission040.csv'
    OUT_PATH = 'submission044.csv'
    
    s39 = pd.read_csv(PATH_M39).sort_values('data_id')
    s40 = pd.read_csv(PATH_M40).sort_values('data_id')
    
    p39 = np.log1p(s39['construction_cost_per_m2_usd'])
    p40 = np.log1p(s40['construction_cost_per_m2_usd'])
    
    # Blend with optimal weight
    # w_opt is weight of M40
    # Blend = (1 - w)*M39 + w*M40
    
    w = max(0.0, min(1.0, w_opt)) # Clamp 0-1
    print(f"Using clamped weight: {w:.4f}")
    
    log_blend = (1 - w) * p39 + w * p40
    final_pred = np.expm1(log_blend)
    
    sub = pd.DataFrame({'data_id': s39['data_id'], 'construction_cost_per_m2_usd': final_pred})
    sub.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    optimize()
