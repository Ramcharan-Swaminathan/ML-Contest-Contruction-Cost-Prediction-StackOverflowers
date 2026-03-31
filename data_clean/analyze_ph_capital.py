import pandas as pd
import os

def analyze_capital():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(SCRIPT_DIR, "train_philippines.csv")
    
    if not os.path.exists(DATA_PATH):
        # Fallback to absolute path if running from root
        DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data_clean", "train_philippines.csv")
        
    df = pd.read_csv(DATA_PATH)
    
    print("Unique Geolocation Names (Encoded):")
    print(df['geolocation_name'].unique()[:10]) # Show first 10
    
    # Use existing column
    if 'is_capital' in df.columns:
        print("Found existing 'is_capital' column.")
        df['is_capital_derived'] = df['is_capital']
    else:
        print("Warning: 'is_capital' not found. Re-derivation impossible on encoded data.")
        return
    
    print("\n--- Stats for is_capital ---")
    print(df['is_capital_derived'].value_counts())
    
    # Compare Target
    target = 'construction_cost_per_m2_usd'
    print(f"\n--- Target Stats ({target}) ---")
    print(df.groupby('is_capital_derived')[target].describe())
    
    # Check Distance Correlation
    if 'straight_distance_to_capital_km' in df.columns:
        corr_dist = df['straight_distance_to_capital_km'].corr(df[target])
        corr_cap = df['is_capital_derived'].corr(df[target])
        print(f"\nCorrelation with Target:")
        print(f"  is_capital: {corr_cap:.4f}")
        print(f"  distance_to_capital: {corr_dist:.4f}")
        
        print("\nDistance stats for Capital vs Non-Capital:")
        print(df.groupby('is_capital_derived')['straight_distance_to_capital_km'].describe())

if __name__ == "__main__":
    analyze_capital()
