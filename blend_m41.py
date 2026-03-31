import pandas as pd
import numpy as np

# --- Config ---
PATH_M40 = 'submission040.csv' # Evolved Scalar (0.21957)
PATH_M21 = 'phase1_archive/submission021.csv' # Visual Champ (0.2205)
OUT_PATH = 'submission041.csv'

def blend():
    print("Loading Submissions...")
    s40 = pd.read_csv(PATH_M40)
    s21 = pd.read_csv(PATH_M21)
    
    # Sort
    s40 = s40.sort_values('data_id').reset_index(drop=True)
    s21 = s21.sort_values('data_id').reset_index(drop=True)
    
    # Check alignment
    if not s40['data_id'].equals(s21['data_id']):
        raise ValueError("Data IDs do not match!")
        
    p40 = np.log1p(s40['construction_cost_per_m2_usd'])
    p21 = np.log1p(s21['construction_cost_per_m2_usd'])
    
    # Correlation Matrix
    corr = np.corrcoef(p40, p21)[0, 1]
    print(f"\nCorrelation M40 (Scalar) vs M21 (Visual): {corr:.5f}")
    
    # Blending Strategy
    # M40 is significantly better than M21 (0.2195 vs 0.2205 is ~0.001 difference).
    # This suggests M40 has "solved" the scalar part better than M21 did visually.
    # But M21 sees things M40 can't (roof texture).
    
    w40 = 0.6
    w21 = 0.4
    
    print(f"Blending: {w40} * M40 + {w21} * M21")
    
    log_blend = w40 * p40 + w21 * p21
    final_pred = np.expm1(log_blend)
    
    # Output
    sub = pd.DataFrame({'data_id': s40['data_id'], 'construction_cost_per_m2_usd': final_pred})
    sub.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    blend()
