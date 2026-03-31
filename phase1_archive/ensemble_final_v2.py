import pandas as pd
import numpy as np

# --- Config ---
# The Champion: LightGBM trained on Pseudo-Labeled Data
FILE_1 = 'submission027.csv' 
WEIGHT_1 = 0.10

# The Challenger: CatBoost trained on Original Data (Diversity)
FILE_2 = 'sub_ens_kestav.csv'
WEIGHT_2 = 0.90

OUTPUT_FILE = 'sub_027_kestav.csv'

def ensemble():
    print("Loading Submissions...")
    try:
        df1 = pd.read_csv(FILE_1)
        df2 = pd.read_csv(FILE_2)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Check alignment
    if len(df1) != len(df2):
        print("Error: Submissions vary in length!")
        return
        
    # Sort to ensure alignment (robustness)
    if 'data_id' in df1.columns:
        df1 = df1.sort_values('data_id').reset_index(drop=True)
        df2 = df2.sort_values('data_id').reset_index(drop=True)

        if not df1['data_id'].equals(df2['data_id']):
             print("Error: IDs do not match!")
             return

    print("Blending...")
    # Log-space blending (geometric mean) is usually better for skewed Targets, 
    # but linear blending is safer/standard. Let's use Linear since we predict cost directly (expm1'd in scripts).
    # Wait, scripts output 'construction_cost_per_m2_usd'.
    # For RMSLE metrics, geometric mean (mean of logs) is theoretically better.
    
    # Let's do Log-Space Blending:
    # 1. Log transform
    log_pred1 = np.log1p(df1['construction_cost_per_m2_usd'])
    log_pred2 = np.log1p(df2['construction_cost_per_m2_usd'])
    
    # 2. Weighted Average
    final_log_pred = (WEIGHT_1 * log_pred1) + (WEIGHT_2 * log_pred2)
    
    # 3. Expm1
    final_pred = np.expm1(final_log_pred)
    
    # Save
    sub = pd.DataFrame()
    if 'data_id' in df1.columns:
        sub['data_id'] = df1['data_id']
    sub['construction_cost_per_m2_usd'] = final_pred
    
    sub.to_csv(OUTPUT_FILE, index=False)
    print(f"Ensemble Saved to {OUTPUT_FILE}")
    print(f"Recipe: {WEIGHT_1} * M13 + {WEIGHT_2} * M15")

if __name__ == "__main__":
    ensemble()
