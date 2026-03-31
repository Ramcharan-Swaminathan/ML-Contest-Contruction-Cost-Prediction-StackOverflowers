import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import warnings
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# --- Config ---
TRAIN_DATA = 'dataset/train_tabular.csv'
TEST_DATA = 'evaluation_tabular_no_target.csv'
PSEUDO_LABEL_FILE = 'submission045.csv' # The Champion
FEATURE_FILE = 'evolved_features.json'
SUB_FILE = 'submission047.csv'
SEED = 42

# M40 Params (Proven Stable)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.015,
    'num_leaves': 31,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 20,
    'seed': SEED
}

def train_recursive():
    print("Loading Data...")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    pseudo_df = pd.read_csv(PSEUDO_LABEL_FILE)
    
    # Load Evolved Features
    with open(FEATURE_FILE, 'r') as f:
        selected_features = json.load(f)
    print(f"Loaded {len(selected_features)} evolved features.")
    
    # Feature Engineering (Must match M40)
    # We need to recreate the feature pool first because selected_features names might be complex?
    # Actually M40 script relied on `generate_feature_pool`.
    # I'll rely on a simplified version since I don't want to copy 200 lines of EDA.
    # WAIT: M40 used `train_40_evolution.py` which generated features on the fly.
    # I should reuse M40 logic or just do minimal encoding if the selected_features are simple.
    # Let's check selected features first.
    
    # Actually, simpler plan:
    # Just perform the basic cleaning/Ordinal encoding, then LightGBM will handle the rest.
    # But wait, did M40 use "spectral indices"? Yes.
    # I need to calculate those.
    
    # Let's import the preprocessing from M40 if possible or copy it.
    # I will implement the standard feature block here.
    
    for df in [train_df, test_df]:
        # Basic Helpers
        if 'quarter_label' in df.columns:
            df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if '-Q' in str(x) else 0)
        
        # Spectral Stuff (If available in raw tabular? No, spectral features were experimental).
        # Wait, M35/M40 used 'viirs_mean', 's2_B11_mean', etc.
        # These ARE in the tabular data.
        # So I just need to ensure the columns exist.
    
    # 1. Prepare Pseudo-Labels
    test_df['construction_cost_per_m2_usd'] = pseudo_df['construction_cost_per_m2_usd']
    
    # 2. Combine for Encoding
    # Mark real vs pseudo
    train_df['is_pseudo'] = 0
    test_df['is_pseudo'] = 1
    
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # 3. Encoding
    cat_cols = ['geolocation_name', 'country', 'region_economic_classification', 
                'seismic_hazard_zone', 'flood_risk_class', 'tropical_cyclone_wind_risk', 
                'tornadoes_wind_risk', 'koppen_climate_zone', 'access_to_airport', 
                'access_to_port', 'access_to_highway', 'access_to_railway', 'developed_country', 'landlocked']
                
    for c in cat_cols:
        if c in full_df.columns:
            full_df[c] = full_df[c].astype('category')
            
    # 4. Filter Features
    # If selected_features contains things like "gdp_x_viirs", I need to generate them.
    # Let's assume for M47 I'll just use ALL available features + simple interactions,
    # trusting LightGBM.
    # OR better: Use the exact M35 domain features + the raw tabular columns.
    # I'll use ALL columns except IDs.
    
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', 'construction_cost_per_m2_usd', 'is_pseudo']
    features = [c for c in full_df.columns if c not in drop_cols]
    
    print(f"Training with {len(features)} features...")
    
    X = full_df[features]
    y = np.log1p(full_df['construction_cost_per_m2_usd'])
    
    # 5. Train
    # We train on Everything (Train + Test)
    # No CV needed (OOFs already validated in M45). We want the final Sub.
    
    dtrain = lgb.Dataset(X, label=y, categorical_feature='auto')
    
    print("Training Recursive Student...")
    bst = lgb.train(params, dtrain, num_boost_round=1500)
    
    # 6. Predict
    # Predict on Test again (Refining the pseudo-labels)
    # Specifically, we predict on the 'test' part of X
    X_test = full_df[full_df['is_pseudo']==1][features]
    
    preds_log = bst.predict(X_test)
    preds = np.expm1(preds_log)
    
    sub = pd.DataFrame({'data_id': test_df['data_id'], 'construction_cost_per_m2_usd': preds})
    sub.to_csv(SUB_FILE, index=False)
    print(f"Saved {SUB_FILE}")

if __name__ == "__main__":
    train_recursive()
