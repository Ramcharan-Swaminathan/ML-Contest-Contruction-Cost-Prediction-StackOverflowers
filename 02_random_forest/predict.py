import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

def predict_rf():
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
        
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Train Model (RF: No Scaling needed)
        print(f"  Training Random Forest for {country}...")
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            verbose_feature_names_out=False
        )
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        model.fit(X_train, y_train_log)
        
        # Save Model
        import joblib
        model_path = os.path.join(SCRIPT_DIR, f"model_{country}.joblib")
        joblib.dump(model, model_path)
        print(f"  Saved model to {model_path}")

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
    predict_rf()
