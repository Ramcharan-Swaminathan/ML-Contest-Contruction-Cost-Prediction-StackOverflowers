import pandas as pd
import numpy as np

def analyze_diff():
    # Load both submissions
    sub_hybrid = pd.read_csv("submission_hybrid.csv")
    sub_048 = pd.read_csv("../submission048.csv")
    
    # Merge
    merged = pd.merge(sub_hybrid, sub_048, on="data_id", suffixes=('_new', '_old'))
    
    # Calculate difference
    merged['diff'] = merged['construction_cost_per_m2_usd_new'] - merged['construction_cost_per_m2_usd_old']
    merged['abs_diff'] = merged['diff'].abs()
    
    print(f"Total rows: {len(merged)}")
    print(f"Mean Abs Diff: {merged['abs_diff'].mean():.4f}")
    
    # We suspect the issue is Japan.
    # We need to identify Japan rows. Hybrid script merged predictors so we don't have country here easily.
    # But we know hybrid used 'old' for Philippines. So difference should be exactly 0 for Philippines.
    
    ph_diff = merged[merged['abs_diff'] < 1e-9]
    jp_diff = merged[merged['abs_diff'] > 1e-9]
    
    print(f"Rows with 0 diff (Philippines): {len(ph_diff)}")
    print(f"Rows with >0 diff (Japan): {len(jp_diff)}")
    
    if len(jp_diff) > 0:
        print("\n--- Japan Differences ---")
        print(jp_diff['abs_diff'].describe())
        print("\nTop 5 biggest changes:")
        print(jp_diff.sort_values('abs_diff', ascending=False).head(5)[['data_id', 'construction_cost_per_m2_usd_old', 'construction_cost_per_m2_usd_new', 'diff']])

        # Check bias
        print(f"\nMean Diff (New - Old): {jp_diff['diff'].mean():.4f}")
        # If Positive, New predicts higher cost.
        
if __name__ == "__main__":
    analyze_diff()
