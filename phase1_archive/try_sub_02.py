# ==========================================
# SOLAFUNE SUBMISSION PIPELINE
# Mentor: Expert ML Engineer
# ==========================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_FILE = 'dataset/train_tabular.csv'  # Your training data
TEST_FILE = 'evaluation_tabular_no_target.csv'    # Your test data (features for submission)
SAMPLE_SUB = 'sample_submission002.csv' # Optional: to ensure ID order
OUTPUT_FILE = 'submission002.csv' 

# ==========================================
# 1. ROBUST PREPROCESSING
# ==========================================
def preprocess_data(df, is_train=True):
    print(f">> Preprocessing {len(df)} rows...")
    
    # 1. Identify ID column to keep for submission
    # We store IDs separately so they don't interfere with training
    ids = df['data_id'].copy() if 'data_id' in df.columns else None
    
    # 2. Drop non-feature columns
    # We drop ID and Image names.
    # Note: We do NOT drop 'country' or geographic info, those are useful features!
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name']
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    
    # 3. Handle Target Variable (Only for Training Data)
    y = None
    if is_train and 'construction_cost_per_m2_usd' in df_clean.columns:
        # Log1p transforms data: log(1 + x) for RMSLE metric
        y = np.log1p(df_clean['construction_cost_per_m2_usd'])
        df_clean = df_clean.drop(columns=['construction_cost_per_m2_usd'])
    
    # 4. AUTOMATIC CATEGORICAL HANDLING
    # Detect object/string columns
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Add specific risk/geographic columns even if they look like numbers
    known_cats = [
        'seismic_hazard_zone', 'flood_risk_class', 
        'tropical_cyclone_wind_risk', 'tornadoes_wind_risk', 
        'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway'
    ]
    for c in known_cats:
        if c in df_clean.columns and c not in cat_cols:
            cat_cols.append(c)

    # Cast to category for LightGBM
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype('category')
        
    return df_clean, y, ids

# ==========================================
# 2. MAIN EXECUTION
# ==========================================

# --- A. Load All Data ---
print("Loading datasets...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Sanity Check: Ensure Train has target
train_df = train_df.dropna(subset=['construction_cost_per_m2_usd'])

# --- B. Process Data ---
# Note: We process them separately but using the same logic
X_train, y_train, _ = preprocess_data(train_df, is_train=True)
X_test, _, test_ids = preprocess_data(test_df, is_train=False)

# align columns: ensure Test has exact same columns as Train
# (LightGBM is strict about column order and count)
# Get common columns
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"\nFinal Feature Count: {len(common_cols)}")
print(f"Features: {common_cols}")

# --- C. Train on FULL Dataset (GPU Enabled) ---
print("\n>> Training on 100% of data (No Validation Split)...")

train_data = lgb.Dataset(X_train, label=y_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'device': 'cuda',          # GPU ENABLED
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1
}

# We train for a fixed number of rounds since we don't have a validation set to stop us.
# Based on previous runs, 2000-3000 is usually a safe "converged" point for this learning rate.
model = lgb.train(
    params,
    train_data,
    num_boost_round=3000, 
    callbacks=[lgb.log_evaluation(period=500)]
)

# --- D. Predict on Test Set ---
print("\n>> Generating Predictions...")
log_preds = model.predict(X_test)

# Reverse Log Transformation (Expm1)
# We trained on log(1+cost), so we reverse with exp(pred) - 1
final_predictions = np.expm1(log_preds)

# --- E. Create Submission File ---
submission = pd.DataFrame({
    'data_id': test_ids,
    'construction_cost_per_m2_usd': final_predictions
})

# Verify formatting matches Solafune requirements
# Usually NO negative costs allowed. Clip to 0 just in case.
submission['construction_cost_per_m2_usd'] = submission['construction_cost_per_m2_usd'].clip(lower=0)

submission.to_csv(OUTPUT_FILE, index=False)
print(f"\nSUCCESS: Submission saved to '{OUTPUT_FILE}'")
print(f"Head of submission:\n{submission.head()}")