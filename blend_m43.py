import pandas as pd
import numpy as np

# --- Config ---
PATH_M39 = 'submission039.csv' # Current Champ (0.2191)
PATH_M40 = 'submission040.csv' # Evolved Scalar (0.21957)
PATH_M37 = 'dataset/../submission037.csv' # Check if exists, actually it was usually output to root.
# Let's assume root.
OUT_PATH = 'submission043.csv'

def blend():
    print("Loading Submissions...")
    s39 = pd.read_csv(PATH_M39)
    s40 = pd.read_csv(PATH_M40)
    
    # Sort
    s39 = s39.sort_values('data_id').reset_index(drop=True)
    s40 = s40.sort_values('data_id').reset_index(drop=True)
    
    p39 = np.log1p(s39['construction_cost_per_m2_usd'])
    p40 = np.log1p(s40['construction_cost_per_m2_usd'])
    
    corr = np.corrcoef(p39, p40)[0, 1]
    print(f"Correlation M39 vs M40: {corr:.5f}")
    
    # Diversity Check
    # M39 = 0.6*M38 + 0.4*M21
    # M40 = Evolved Scalar
    # M38 and M40 are correlated but distinct (Distilled vs Hard Targets).
    
    # Weights
    # M39 is significantly better (0.2191 vs 0.2196).
    # We don't want to ruin M39. We just want to nudge it.
    w39 = 0.8
    w40 = 0.2
    
    print(f"Blending: {w39} * M39 + {w40} * M40")
    
    log_blend = w39 * p39 + w40 * p40
    final_pred = np.expm1(log_blend)
    
    sub = pd.DataFrame({'data_id': s39['data_id'], 'construction_cost_per_m2_usd': final_pred})
    sub.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    blend()
