import pandas as pd
import numpy as np
import os
from autogluon.tabular import TabularPredictor
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
TRAIN_DATA = 'dataset/train_tabular.csv'
TRAIN_FEATS = 'dataset/image_features_train.csv'
OUTPUT_PATH = 'autogluon_m30_tabular'
TIME_LIMIT = 3600 # 1 Hour

def train_autogluon():
    print("Loading Data...")
    train_df = pd.read_csv(TRAIN_DATA)
    
    if os.path.exists(TRAIN_FEATS):
        print(f"Merging with features from {TRAIN_FEATS}...")
        feats = pd.read_csv(TRAIN_FEATS)
        train_df = train_df.merge(feats, on='data_id', how='left')
    else:
        print("Warning: Image features not found!")

    # 1. Log Transform Target (Standard way to optimize RMSLE in AutoGluon)
    target_col = 'log_cost'
    train_df[target_col] = np.log1p(train_df['construction_cost_per_m2_usd'])
    
    # 2. Drop Leakage Columns & Irrelevant
    drop_cols = ['construction_cost_per_m2_usd', 'data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label']
    train_data = train_df.drop(columns=drop_cols, errors='ignore')
    
    # 3. Train
    print("Initializing TabularPredictor (Metric: RMSE on Log-Target = RMSLE)...")
    predictor = TabularPredictor(
        label=target_col,
        problem_type='regression',
        eval_metric='root_mean_squared_error',
        path=OUTPUT_PATH
    )
    
    print("Fitting Model (Budget: 1 Hour)...")
    predictor.fit(
        train_data,
        time_limit=TIME_LIMIT,
        presets='best_quality' 
    )
    
    print(f"Training Complete. Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    train_autogluon()
