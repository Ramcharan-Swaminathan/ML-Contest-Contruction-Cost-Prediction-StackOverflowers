import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb

# Reuse existing preprocessing classes/functions for Japan
class SmoothedTargetEncoder:
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.means = {}
        self.global_mean = None

    def fit(self, X, y):
        self.global_mean = y.mean()
        stats = y.groupby(X).agg(['count', 'mean'])
        smoothing = self.smoothing
        self.means = (stats['count'] * stats['mean'] + smoothing * self.global_mean) / (stats['count'] + smoothing)
        return self

    def transform(self, X):
        return X.map(self.means).fillna(self.global_mean)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

def process_data(df_train, df_eval, country_name):
    # Same logic as predict.py (simplified as we only need this for Japan mostly)
    print(f"\nProcessing {country_name}...")
    
    # 1. Feature Engineering: Interaction & Capital
    def create_interaction(df):
        if country_name == 'Japan':
            is_capital_mask = df['geolocation_name'].astype(str).str.contains('Tokyo', case=False, na=False)
        else: 
            is_capital_mask = df['geolocation_name'].astype(str).str.contains('National Capital Region', case=False, na=False)
        
        df['is_capital'] = is_capital_mask.astype(int)
        
        if 'us_cpi' in df.columns:
            df['us_cpi_x_is_capital'] = df['us_cpi'] * df['is_capital']
        return df

    df_train = create_interaction(df_train.copy())
    df_eval = create_interaction(df_eval.copy())
    
    # 2. Impute: Tropical Cyclone Wind Risk
    wind_col = 'tropical_cyclone_wind_risk'
    if wind_col in df_train.columns:
        mode_val = df_train[wind_col].mode()[0]
        df_train[wind_col] = df_train[wind_col].fillna(mode_val)
        if wind_col in df_eval.columns:
            df_eval[wind_col] = df_eval[wind_col].fillna(mode_val)
    
    # 3. Target Encoding
    target_col = 'construction_cost_per_m2_usd'
    encoder = SmoothedTargetEncoder(smoothing=10)
    df_train['geolocation_name'] = encoder.fit_transform(df_train['geolocation_name'], df_train[target_col])
    df_eval['geolocation_name'] = encoder.transform(df_eval['geolocation_name'])
    
    # 4. Drop Noise
    drop_cols = ['country', 'data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'deflated_gdp_usd', 'year', 'access_to_highway']
    
    cols_to_drop_train = [c for c in drop_cols if c in df_train.columns]
    cols_to_drop_eval = [c for c in drop_cols if c in df_eval.columns]
    
    df_train = df_train.drop(columns=cols_to_drop_train)
    df_eval = df_eval.drop(columns=cols_to_drop_eval)
    
    return df_train, df_eval

def predict_hybrid():
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    TRAIN_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_tabular.csv")
    EVAL_PATH = os.path.join(PROJECT_ROOT, "evaluation_dataset", "evaluation_tabular_no_target.csv")
    SUBMISSION_PATH = os.path.join(SCRIPT_DIR, "submission_hybrid.csv") # Save as hybrid to avoid overwriting default for now
    
    # Pre-computed Philippines path
    # User specified: @[submission048.csv] which is usually in root or specified path
    # Try finding it in project root
    PH_PREDS_PATH = os.path.join(PROJECT_ROOT, "submission048.csv")
    
    print("Loading datasets...")
    df_train_full = pd.read_csv(TRAIN_PATH)
    df_eval_full = pd.read_csv(EVAL_PATH)
    
    df_preds_list = []
    
    # JAPAN: Predict with LightGBM
    print("\n--- JAPAN: LightGBM Prediction ---")
    country = 'Japan'
    df_train_c = df_train_full[df_train_full['country'] == country].copy()
    df_eval_c = df_eval_full[df_eval_full['country'] == country].copy()
    eval_ids_c = df_eval_c['data_id'].values
    
    # Process
    df_train_proc, df_eval_proc = process_data(df_train_c, df_eval_c, country)
    
    # Drop target
    target_col = 'construction_cost_per_m2_usd'
    X_eval = df_eval_proc
    
    # Categoricals
    obj_cols = X_eval.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        X_eval[col] = X_eval[col].astype('category')
    
    # Load Model
    model_path = os.path.join(SCRIPT_DIR, f"model_{country}_train.joblib")
    print(f"  Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
        
        # Predict
        print(f"  Predicting for {country}...")
        y_pred_log = model.predict(X_eval)
        y_pred = np.expm1(y_pred_log)
        
        df_pred_jp = pd.DataFrame({
            'data_id': eval_ids_c,
            'construction_cost_per_m2_usd': y_pred
        })
        df_preds_list.append(df_pred_jp)
        print(f"  Generated {len(df_pred_jp)} predictions for Japan.")
        
    except FileNotFoundError:
        print(f"  [ERROR] Model not found at {model_path}.")
        return

    # PHILIPPINES: Load from submission048.csv
    print("\n--- PHILIPPINES: Loading from submission048.csv ---")
    country = 'Philippines'
    df_eval_ph = df_eval_full[df_eval_full['country'] == country].copy()
    eval_ids_ph = df_eval_ph['data_id'].values
    
    if os.path.exists(PH_PREDS_PATH):
        print(f"  Reading {PH_PREDS_PATH}...")
        df_old_sub = pd.read_csv(PH_PREDS_PATH)
        
        # Filter for Philippines IDs
        # We merge on data_id to get the values for the specific PH eval set
        # This implicitly filters
        df_pred_ph = df_old_sub[df_old_sub['data_id'].isin(eval_ids_ph)].copy()
        
        # Check completeness
        if len(df_pred_ph) != len(eval_ids_ph):
            print(f"  [WARNING] Mismatch in count! Needed {len(eval_ids_ph)}, found {len(df_pred_ph)}.")
            # If mismatch, we might need to fill? Or just error?
            # Let's assume strict match is desired.
            missing_ids = set(eval_ids_ph) - set(df_pred_ph['data_id'])
            if missing_ids:
                print(f"  Missing IDs: {missing_ids}")
        
        df_preds_list.append(df_pred_ph)
        print(f"  Loaded {len(df_pred_ph)} predictions for Philippines.")
        
    else:
        print(f"  [ERROR] File {PH_PREDS_PATH} not found!")
        return

    # Combine
    submission_df = pd.concat(df_preds_list)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved HYBRID submission to {SUBMISSION_PATH}")
    print(submission_df.head())

if __name__ == "__main__":
    predict_hybrid()
