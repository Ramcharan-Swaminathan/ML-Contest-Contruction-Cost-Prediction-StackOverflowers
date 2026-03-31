
import pandas as pd
import numpy as np

# --- Config ---
PATH_M21 = './phase1_archive/submission021.csv'
PATH_M35 = './phase4_archive/submission035.csv'
OUT_PATH = 'submission037.csv'

# Weights based on performance (M21=0.2205, M35=0.2291)
# M21 is significantly better, so it gets the lion's share.
# But M35 captures fundamentals.
w_m21 = 0.75
w_m35 = 0.25

def blend():
    print("Loading Submissions...")
    s21 = pd.read_csv(PATH_M21)
    s35 = pd.read_csv(PATH_M35)
    
    # Sort by data_id to ensure alignment
    s21 = s21.sort_values('data_id').reset_index(drop=True)
    s35 = s35.sort_values('data_id').reset_index(drop=True)
    
    # Check alignment
    if not s21['data_id'].equals(s35['data_id']):
        raise ValueError("Data IDs do not match!")
        
    p21 = s21['construction_cost_per_m2_usd']
    p35 = s35['construction_cost_per_m2_usd']
    
    # Correlation Analysis
    # Log-transform before correlation because the metric is RMSLE
    log_p21 = np.log1p(p21)
    log_p35 = np.log1p(p35)
    
    corr = np.corrcoef(log_p21, log_p35)[0, 1]
    print(f"Correlation between M21 (Visual) and M35 (Scalar): {corr:.4f}")
    
    if corr > 0.99:
        print("Warning: Models are extremely correlated. Blending may produce minimal gains.")
    elif corr < 0.95:
        print("Great! Models have sufficient diversity. Blending should work well.")
        
    # Blending (Geometric Mean is often better for Log-Target, but Arithmetic is safer)
    # Let's use Arithmetic on the Log scale (which is Geometric on the Raw scale).
    # This aligns with minimizing RMSLE.
    log_blend = w_m21 * log_p21 + w_m35 * log_p35
    final_pred = np.expm1(log_blend)
    
    # Output
    sub = pd.DataFrame({'data_id': s21['data_id'], 'construction_cost_per_m2_usd': final_pred})
    sub.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} with weights: M21={w_m21}, M35={w_m35}")

if __name__ == "__main__":
    blend()
