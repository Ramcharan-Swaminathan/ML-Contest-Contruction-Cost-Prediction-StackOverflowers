import pandas as pd
import numpy as np

# --- Config ---
PATH_M45 = 'submission045.csv'
OUT_PATH = 'submission046.csv'
BIAS = -0.001577 # Measured from M40 OOF

def correct_bias():
    print(f"Applying Bias Correction: +{-BIAS:.6f}")
    df = pd.read_csv(PATH_M45)
    
    # Log Transform
    log_pred = np.log1p(df['construction_cost_per_m2_usd'])
    
    # Correct Bias (If model underpredicts, we add)
    # Bias = Pred - Target = -0.0015
    # Target approx Pred - Bias = Pred + 0.0015
    corrected_log = log_pred - BIAS
    
    final_pred = np.expm1(corrected_log)
    
    df['construction_cost_per_m2_usd'] = final_pred
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    correct_bias()
