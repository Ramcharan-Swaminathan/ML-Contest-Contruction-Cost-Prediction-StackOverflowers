import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import argparse

def predict(args):
    # Paths
    # Using defaults or args
    # When using argparse, defaults in add_argument take care of it if args is populated
    # But if called with empty string or None, we need fallback
    if isinstance(args, argparse.Namespace):
        model_path = args.model_path 
        data_path = args.data_path
        output_path = args.output_path
    else:
         model_path = 'model_checkpoint.txt'
         data_path = 'evaluation_tabular_no_target.csv'
         output_path = 'submission001.csv'
         
    # Fallback if None (e.g. required arg not passed but we are in dev/test mode)
    if data_path is None:
        data_path = 'evaluation_tabular_no_target.csv'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data not found at {data_path}")

    print(f"Loading model from {model_path}...")
    bst = lgb.Booster(model_file=model_path)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Store IDs for submission if present
    ids = None
    if 'data_id' in df.columns:
        ids = df['data_id']

    # --- Preprocessing (Must match train.py) ---
    print("Preprocessing data...")
    
    # 1. Feature Engineering: Extract Quarter
    if 'quarter_label' in df.columns:
        df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if isinstance(x, str) and '-Q' in x else 0)
    
    # 2. Drop columns
    # Note: geolocation_name is KEPT (removed from drop list compared to earlier attempts)
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', 'construction_cost_per_m2_usd'] 
    
    # Only drop columns that actually exist in the test dataframe
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # 3. Handle Categorical Columns
    # Using dynamic detection to match train.py exactly
    # 3. Handle Categorical Columns
    # Using dynamic detection can be risky if order or types differ. 
    # LightGBM validation is strict on input Dataframe column names and types.
    p_cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Detected categorical columns: {p_cat_cols}")
    
    for col in p_cat_cols:
        X[col] = X[col].astype('category')
    
    print(f"Predicting on {X.shape[0]} rows and {X.shape[1]} features...")

    # Validate feature names against model
    # bst.feature_name() returns the list of features the model was trained on
    model_features = bst.feature_name()
    
    # Ensure all model features exist in X and are in the same ORDER
    # This is crucial for LightGBM consistency
    missing_features = [f for f in model_features if f not in X.columns]
    if missing_features:
        raise ValueError(f"Input data is missing features expected by model: {missing_features}")
        
    # Reorder X columns to match model exactly
    X = X[model_features]
    print("Features reordered to match model.") 
    
    # --- Prediction ---
    # Log-scale predictions
    preds_log = bst.predict(X)
    
    # Inverse transform (Log -> Actual)
    preds_actual = np.expm1(preds_log)
    
    # --- Save Submission ---
    submission = pd.DataFrame()
    if ids is not None:
        submission['data_id'] = ids
    
    submission['construction_cost_per_m2_usd'] = preds_actual
    
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions using trained LightGBM model")
    parser.add_argument('--data_path', type=str, help='Path to evaluation/test csv file')
    parser.add_argument('--model_path', type=str, default='model_checkpoint.txt', help='Path to trained model file')
    parser.add_argument('--output_path', type=str, default='submission.csv', help='Path to save predictions')
    
    args = parser.parse_args()
    # Support running without args for test
    if args.data_path is None:
        # Fallback for testing
        pass 
    
    predict(args)
