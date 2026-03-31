import pandas as pd
import numpy as np
import os

class SmoothedTargetEncoder:
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.means = {}
        self.global_mean = None

    def fit(self, X, y):
        self.global_mean = y.mean()
        stats = y.groupby(X).agg(['count', 'mean'])
        smoothing = self.smoothing
        
        # Calculate smoothed means
        # Formula: (n * mean + m * global_mean) / (n + m)
        rho = 1 / (1 + np.exp(-(stats['count'] - smoothing))) # Wait, simpler formula is better for clarity?
        # Standard smoothing formula often used:
        # smoothed = (count * mean + smoothing * global_mean) / (count + smoothing)
        
        # Let's use the standard one mentioned in typical kaggle discussions
        self.means = (stats['count'] * stats['mean'] + smoothing * self.global_mean) / (stats['count'] + smoothing)
        return self

    def transform(self, X):
        return X.map(self.means).fillna(self.global_mean)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

def clean_data():
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_tabular.csv")
    OUTPUT_DIR = SCRIPT_DIR
    
    # 1. Load Data
    print(f"Loading data from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    # 2. Split Data: Japan vs. Philippines
    print("Splitting data into Japan and Philippines...")
    df_jp = df[df['country'] == 'Japan'].copy()
    df_ph = df[df['country'] == 'Philippines'].copy()
    
    print(f"Japan samples: {len(df_jp)}")
    print(f"Philippines samples: {len(df_ph)}")
    
    # Process each dataset
    for name, data in [('Japan', df_jp), ('Philippines', df_ph)]:
        print(f"\nProcessing {name}...")
        
        # 3. Impute: Fill missing wind_risk values with Mode
        if 'tropical_cyclone_wind_risk' in data.columns:
            missing_wind = data['tropical_cyclone_wind_risk'].isnull().sum()
            print(f"  Missing tropical_cyclone_wind_risk: {missing_wind}")
            if missing_wind > 0:
                mode_val = data['tropical_cyclone_wind_risk'].mode()[0]
                data['tropical_cyclone_wind_risk'] = data['tropical_cyclone_wind_risk'].fillna(mode_val)
                print(f"  Filled missing tropical_cyclone_wind_risk with mode: {mode_val}")
        
        # 4. Target Encode: Apply SmoothedTargetEncoder to geolocation_name
        target_col = 'construction_cost_per_m2_usd'
        if 'geolocation_name' in data.columns:
            print("  Applying SmoothedTargetEncoder to geolocation_name...")
            encoder = SmoothedTargetEncoder(smoothing=10)
            # Create a new column or replace? Request said "Apply... to geolocation_name"
            # Typically means replacing or creating a new feature. I will replace it to make it "encoded"
            # BUT usually we keep original or create new. Let's create 'geolocation_name_encoded' for safety, 
            # or given "Apply... to", I can just replace it if the user wants 'clean' data for modeling.
            # "Target Encode: Apply SmoothedTargetEncoder to geolocation_name." -> usually implies transformation.
            # I will overwrite 'geolocation_name' with the encoded values to prepare purely for modeling, 
            # but usually it's better to keep original for reference... 
            # Reviewing "Drop Noise"... geolocation_name is NOT in drop noise list.
            # I will overwrite it to be safe for a "model-ready" dataset, as textual geolocation isn't useful for models directly.
            data['geolocation_name'] = encoder.fit_transform(data['geolocation_name'], data[target_col])
        
        # 5. Create Interaction: Multiply us_cpi * is_capital
        print("  Creating interaction features...")
        is_capital = pd.Series(False, index=data.index)
        
        if name == 'Japan':
            is_capital = data['geolocation_name'].astype(str).str.contains('Tokyo', case=False, na=False) 
            # Wait, I just target encoded geolocation_name! I CANNOT check string AFTER encoding.
            # Logic error in plan sequence. Interaction must happen BEFORE encoding or use a temporary column.
            # I must fix this order.
        elif name == 'Philippines':
             # Note: geolocation_name might be encoded now. Need to fix order.
             # Moving Interaction BEFORE Target Encoding in the logic below.
             pass
             
    # RE-DO PROPER ORDER LOGIC
    # Re-writing the loop logic to be correct.
    
    pass

def process_dataframe(df, country_name):
    print(f"\nProcessing {country_name}...")
    
    # 5. Create Interaction (Moved before Encoding because it relies on string value)
    # "Multiply us_cpi * is_capital (create a boolean column for Tokyo/Manila)."
    print("  Creating interaction features...")
    
    # Identify Capital
    # Japan: Tokyo
    # Philippines: National Capital Region
    if country_name == 'Japan':
        is_capital_mask = df['geolocation_name'].astype(str).str.contains('Tokyo', case=False, na=False)
    else: # Philippines
        is_capital_mask = df['geolocation_name'].astype(str).str.contains('National Capital Region', case=False, na=False)
        
    df['is_capital'] = is_capital_mask.astype(int)
    
    # us_cpi * is_capital
    if 'us_cpi' in df.columns:
        df['us_cpi_x_is_capital'] = df['us_cpi'] * df['is_capital']
    
    # 3. Impute: Fill missing tropical_cyclone_wind_risk values with Mode
    if 'tropical_cyclone_wind_risk' in df.columns:
        missing_wind = df['tropical_cyclone_wind_risk'].isnull().sum()
        if missing_wind > 0:
            mode_val = df['tropical_cyclone_wind_risk'].mode()[0]
            df['tropical_cyclone_wind_risk'] = df['tropical_cyclone_wind_risk'].fillna(mode_val)
            print(f"  Filled {missing_wind} missing tropical_cyclone_wind_risk values with mode: {mode_val}")

    # 4. Target Encode: Apply SmoothedTargetEncoder to geolocation_name
    target_col = 'construction_cost_per_m2_usd'
    if 'geolocation_name' in df.columns:
        print("  Applying SmoothedTargetEncoder to geolocation_name...")
        encoder = SmoothedTargetEncoder(smoothing=10)
        # Replacing the column with encoded values
        df['geolocation_name'] = encoder.fit_transform(df['geolocation_name'], df[target_col])

    # 6. Drop Noise
    drop_cols = ['country', 'data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'deflated_gdp_usd', 'year', 'access_to_highway']
    # Also drop 'is_capital' intermediate if not requested? User asked "create a boolean column...", so keep it.
    
    print(f"  Dropping noise columns: {drop_cols}")
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    if len(existing_drop_cols) < len(drop_cols):
        print(f"  Warning: Some columns to drop were not found: {set(drop_cols) - set(existing_drop_cols)}")
        
    df = df.drop(columns=existing_drop_cols)
    
    # Drop Capital Features for Philippines (As per user request due to high redundancy)
    if country_name == 'Philippines':
        drop_cols_ph = ['is_capital', 'us_cpi_x_is_capital']
        df = df.drop(columns=drop_cols_ph, errors='ignore')
        print(f"  Dropped special features for Philippines: {drop_cols_ph}")
    
    return df

if __name__ == "__main__":
    # Load Data
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_tabular.csv")
    
    df = pd.read_csv(DATASET_PATH)
    
    # Split
    df_jp = df[df['country'] == 'Japan'].copy()
    df_ph = df[df['country'] == 'Philippines'].copy()
    
    # Process
    df_jp_clean = process_dataframe(df_jp, 'Japan')
    df_ph_clean = process_dataframe(df_ph, 'Philippines')
    
    # Store
    jp_out = os.path.join(SCRIPT_DIR, "train_japan.csv")
    ph_out = os.path.join(SCRIPT_DIR, "train_philippines.csv")
    
    df_jp_clean.to_csv(jp_out, index=False)
    df_ph_clean.to_csv(ph_out, index=False)
    
    print(f"\nSaved Japan data to: {jp_out}")
    print(f"Saved Philippines data to: {ph_out}")
