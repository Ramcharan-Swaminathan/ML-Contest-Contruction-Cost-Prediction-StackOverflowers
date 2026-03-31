import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# --- Config ---
FILE_M21 = 'submission021.csv' # Deep Visuals (0.2205)
FILE_M24 = 'submission024.csv' # Explicit Stats (0.2210)
OUTPUT_FILE = 'submission029.csv'

def blend_specialists():
    print("Loading Submissions...")
    try:
        df_21 = pd.read_csv(FILE_M21)
        df_24 = pd.read_csv(FILE_M24)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # sort by data_id to ensure alignment
    df_21 = df_21.sort_values('data_id').reset_index(drop=True)
    df_24 = df_24.sort_values('data_id').reset_index(drop=True)
    
    # Check alignment
    if not df_21['data_id'].equals(df_24['data_id']):
        print("Error: data_id mismatch!")
        return

    p21 = df_21['construction_cost_per_m2_usd'].values
    p24 = df_24['construction_cost_per_m2_usd'].values
    
    # Log transform for correlation analysis (more Gaussian)
    log_p21 = np.log1p(p21)
    log_p24 = np.log1p(p24)
    
    corr, _ = pearsonr(log_p21, log_p24)
    print(f"Correlation between M21 and M24: {corr:.5f}")
    
    # Blend (Equal Weights for variance reduction)
    # Blending in Log space is geometric mean, Linear space is arithmetic.
    # Usually arithmetic on the final preds is safest for RMSLE unless we are very sure.
    # Let's do simple arithmetic blend.
    
    p_blend = 0.5 * p21 + 0.5 * p24
    
    sub = pd.DataFrame()
    sub['data_id'] = df_21['data_id']
    sub['construction_cost_per_m2_usd'] = p_blend
    
    print(f"Saving blend to {OUTPUT_FILE}...")
    sub.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    blend_specialists()
