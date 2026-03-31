import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def train(args):
    # Set random seed
    np.random.seed(42)

    device = 'cuda' # Default to cuda as per notebook
    print(f"Training will attempt to use: {device}")

    # 1. Load Dataset
    data_path = 'dataset/train_tabular.csv'
    target_col = 'construction_cost_per_m2_usd'

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {data_path} with shape {df.shape}")
    else:
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # 3. Preprocessing
    # Drop high-cardinality identifier columns
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name']
    # Removing these cols from drop list if they don't exist, just in case
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    # Feature Engineering: Extract Quarter
    if 'quarter_label' in df.columns:
        # Format is 'YYYY-QX', we want the X.
        df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if isinstance(x, str) and '-Q' in x else 0)
        drop_cols.append('quarter_label')

    X = df.drop(columns=drop_cols + [target_col])
    # Apply Log Transform for RMSLE optimization
    y = np.log1p(df[target_col])

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {cat_cols}")

    for col in cat_cols:
        X[col] = X[col].astype('category')

    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_valid.shape}")

    # 4. Training Configuration
    # Merging default params with args
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': device,
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'learning_rate': args.learning_rate,
        'num_leaves': args.num_leaves,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_child_samples': args.min_child_samples,
        'reg_alpha': args.reg_alpha,
        'reg_lambda': args.reg_lambda,
        'max_depth': args.max_depth,
        'colsample_bytree': 0.8, # Subsample features
        'feature_fraction': 0.8,
    }
    
    # Dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=cat_cols)

    num_boost_round = 1000
    early_stopping_rounds = 50

    print("Starting training...")
    try:
        bst = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds), lgb.log_evaluation(period=50)]
        )
        
        print("Training completed.")
        
        # Save model
        bst.save_model('model_checkpoint.txt')
        print(f"Model saved to model_checkpoint.txt")
        
        # Predict on validation set
        val_preds_log = bst.predict(X_valid, num_iteration=bst.best_iteration)
        
        # Convert back to actual values (USD)
        val_preds_actual = np.expm1(val_preds_log)
        y_valid_actual = np.expm1(y_valid)
        
        # Calculate RMSLE using actual values
        val_rmsle = np.sqrt(mean_squared_error(np.log1p(y_valid_actual), np.log1p(val_preds_actual)))
        print(f"Validation RMSLE (calculated on actuals): {val_rmsle}")
        
        # Training scores for checking overfitting
        train_preds_log = bst.predict(X_train, num_iteration=bst.best_iteration)
        train_preds_actual = np.expm1(train_preds_log)
        y_train_actual = np.expm1(y_train)
        
        train_rmsle = np.sqrt(mean_squared_error(np.log1p(y_train_actual), np.log1p(train_preds_actual)))
        print(f"Training RMSLE (calculated on actuals): {train_rmsle}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        # CPU Fallback logic could go here if needed, but keeping it simple for now as we just want to reproduce

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    
    # Defaults tuned to reduce overfitting
    parser.add_argument('--learning_rate', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--num_leaves', type=int, default=31, help='Number of leaves')
    parser.add_argument('--min_child_samples', type=int, default=30, help='Min child samples (default 30)')
    parser.add_argument('--reg_alpha', type=float, default=0.1, help='L1 regularization')
    parser.add_argument('--reg_lambda', type=float, default=0.1, help='L2 regularization')
    parser.add_argument('--max_depth', type=int, default=-1, help='Max depth')

    args = parser.parse_args()
    train(args)
