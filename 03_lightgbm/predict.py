import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import joblib

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
    print(f"\nProcessing {country_name}...")
    
    # 1. Feature Engineering: Interaction & Capital
    def create_interaction(df):
        if country_name == 'Japan':
            is_capital_mask = df['geolocation_name'].astype(str).str.contains('Tokyo', case=False, na=False)
        else: # Philippines
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
        print(f"  Imputing {wind_col}...")
        mode_val = df_train[wind_col].mode()[0]
        df_train[wind_col] = df_train[wind_col].fillna(mode_val)
        if wind_col in df_eval.columns:
            df_eval[wind_col] = df_eval[wind_col].fillna(mode_val)
    
    # 3. Target Encoding
    print("  Applying SmoothedTargetEncoder...")
    target_col = 'construction_cost_per_m2_usd'
    encoder = SmoothedTargetEncoder(smoothing=10)
    df_train['geolocation_name'] = encoder.fit_transform(df_train['geolocation_name'], df_train[target_col])
    
    # Apply to Eval
    df_eval['geolocation_name'] = encoder.transform(df_eval['geolocation_name'])
    
    # 4. Drop Noise
    drop_cols = ['country', 'data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'deflated_gdp_usd', 'year', 'access_to_highway']
    
    cols_to_drop_train = [c for c in drop_cols if c in df_train.columns]
    cols_to_drop_eval = [c for c in drop_cols if c in df_eval.columns]
    
    df_train = df_train.drop(columns=cols_to_drop_train)
    df_eval = df_eval.drop(columns=cols_to_drop_eval)
    
    return df_train, df_eval

def predict_lgbm():
    # Verify GPU
    try:
        print("\nVerifying LightGBM GPU support...")
        from lightgbm import train as lgb_train, Dataset as lgb_Dataset
        # Dummy train
        d = lgb_Dataset(np.array([[1,2],[2,3]]), label=np.array([0,1]))
        p = {'device': 'cuda', 'verbosity': -1, 'min_data_in_leaf': 1}
        lgb_train(p, d, num_boost_round=1)
        print("  [SUCCESS] LightGBM initialized with device='cuda' without error.")
    except Exception as e:
        print(f"  [WARNING] LightGBM failed to use GPU: {e}")
        print("  Please ensure lightgbm is installed with GPU support (pip install check).")

    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    TRAIN_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_tabular.csv")
    EVAL_PATH = os.path.join(PROJECT_ROOT, "evaluation_dataset", "evaluation_tabular_no_target.csv")
    SUBMISSION_PATH = os.path.join(SCRIPT_DIR, "submission.csv")
    
    print("Loading datasets...")
    df_train_full = pd.read_csv(TRAIN_PATH)
    df_eval_full = pd.read_csv(EVAL_PATH)
    
    eval_ids = df_eval_full['data_id'].copy()
    
    df_preds_list = []
    
    # Tuned Params (Copied from train_lgbm.py)
    params_japan = {
        'learning_rate': 0.01563892224383956,
        'num_leaves': 22,
        'max_depth': 5,
        'min_child_samples': 11,
        'subsample': 0.8010934630069068,
        'colsample_bytree': 0.830133521161127,
        'reg_alpha': 0.004931727017405291,
        'reg_lambda': 1.3691380094884889e-08,
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'device': 'cuda'
    }

    params_philippines = {
        'learning_rate': 0.021026032262560463,
        'num_leaves': 136,
        'max_depth': 13,
        'min_child_samples': 72,
        'subsample': 0.7410531102911122,
        'colsample_bytree': 0.925811310318166,
        'reg_alpha': 4.002456541286607e-07,
        'reg_lambda': 6.030902568206356e-07,
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'device': 'cuda'
    }
    
    params_map = {'Japan': params_japan, 'Philippines': params_philippines}
    
    for country in ['Japan', 'Philippines']:
        df_train_c = df_train_full[df_train_full['country'] == country].copy()
        df_eval_c = df_eval_full[df_eval_full['country'] == country].copy()
        
        eval_ids_c = df_eval_c['data_id'].values
        
        # Process
        df_train_proc, df_eval_proc = process_data(df_train_c, df_eval_c, country)
        
        # Prepare for Training
        target_col = 'construction_cost_per_m2_usd'
        y_train = df_train_proc[target_col]
        X_train = df_train_proc.drop(columns=[target_col])
        X_eval = df_eval_proc
        
        # Log Transform
        y_train_log = np.log1p(y_train)
        
        # Categoricals
        obj_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        for col in obj_cols:
            X_train[col] = X_train[col].astype('category')
            if col in X_eval.columns:
                X_eval[col] = X_eval[col].astype('category')
        
        # Load Model
        model_path = os.path.join(SCRIPT_DIR, f"model_{country}_train.joblib")
        print(f"  Loading model from {model_path}...")
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"  [ERROR] Model not found at {model_path}. Please run training script first.")
            continue
        
        # Predict
        print(f"  Predicting for {country}...")
        y_pred_log = model.predict(X_eval)
        y_pred = np.expm1(y_pred_log)
        
        df_pred = pd.DataFrame({
            'data_id': eval_ids_c,
            'construction_cost_per_m2_usd': y_pred
        })
        df_preds_list.append(df_pred)

    submission_df = pd.concat(df_preds_list)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved submission to {SUBMISSION_PATH}")
    print(submission_df.head())

if __name__ == "__main__":
    predict_lgbm()
