import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import argparse
import rasterio
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def extract_image_features(df, image_dir):
    """
    Extracts statistical features from Sentinel-2 and VIIRS images.
    """
    print("Extracting image features (this might take a moment)...")
    
    # Initialize lists to store features
    # Sentinel-2: 12 bands. We'll compute Mean and Std for each.
    # Bands: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
    s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    s2_features = []
    
    # VIIRS: 1 band (avg_rad). Mean, Max, Std.
    viirs_features = []
    
    total = len(df)
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{total}")
            
        # --- Sentinel-2 ---
        s2_filename = row.get('sentinel2_tiff_file_name')
        s2_path = os.path.join(image_dir, s2_filename) if isinstance(s2_filename, str) else None
        
        s2_stats = {}
        if s2_path and os.path.exists(s2_path):
            try:
                with rasterio.open(s2_path) as src:
                    # Read all bands (channels, height, width)
                    img = src.read() 
                    # img shape: (12, H, W)
                    
                    for i, band_name in enumerate(s2_bands):
                        band_data = img[i, :, :]
                        # Mask 0 or nodata if necessary, but "median composite" usually implies valid data
                        # We'll just take stats of the whole chip
                        s2_stats[f's2_{band_name}_mean'] = np.nanmean(band_data)
                        s2_stats[f's2_{band_name}_std'] = np.nanstd(band_data)
                        s2_stats[f's2_{band_name}_max'] = np.nanmax(band_data)
            except Exception as e:
                print(f"Error reading Sentinel-2 {s2_path}: {e}")
                # Fill with NaN (LGBM handles it)
                for band_name in s2_bands:
                    s2_stats[f's2_{band_name}_mean'] = np.nan
                    s2_stats[f's2_{band_name}_std'] = np.nan
                    s2_stats[f's2_{band_name}_max'] = np.nan
        else:
             for band_name in s2_bands:
                    s2_stats[f's2_{band_name}_mean'] = np.nan
                    s2_stats[f's2_{band_name}_std'] = np.nan
                    s2_stats[f's2_{band_name}_max'] = np.nan
        
        s2_features.append(s2_stats)
        
        # --- VIIRS ---
        viirs_filename = row.get('viirs_tiff_file_name')
        viirs_path = os.path.join(image_dir, viirs_filename) if isinstance(viirs_filename, str) else None
        
        viirs_stats = {}
        if viirs_path and os.path.exists(viirs_path):
            try:
                with rasterio.open(viirs_path) as src:
                    img = src.read(1) # Only 1 band
                    viirs_stats['viirs_mean'] = np.nanmean(img)
                    viirs_stats['viirs_max'] = np.nanmax(img)
                    viirs_stats['viirs_std'] = np.nanstd(img)
            except Exception as e:
                 print(f"Error reading VIIRS {viirs_path}: {e}")
                 viirs_stats['viirs_mean'] = np.nan
                 viirs_stats['viirs_max'] = np.nan
                 viirs_stats['viirs_std'] = np.nan
        else:
             viirs_stats['viirs_mean'] = np.nan
             viirs_stats['viirs_max'] = np.nan
             viirs_stats['viirs_std'] = np.nan
        
        viirs_features.append(viirs_stats)

    # Convert to DataFrames and concat
    s2_df = pd.DataFrame(s2_features, index=df.index)
    viirs_df = pd.DataFrame(viirs_features, index=df.index)
    
    return pd.concat([df, s2_df, viirs_df], axis=1)

def train(args):
    np.random.seed(42)
    device = 'cuda'
    
    # 1. Load Data
    data_path = 'dataset/train_tabular.csv'
    image_dir = 'dataset/train_composite'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")

    df = pd.read_csv(data_path)
    
    # 2. Extract Image Features (Method 03 Addition)
    df = extract_image_features(df, image_dir)
    
    # 3. Preprocessing (Standard)
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name']
    
    # Quarter Extraction
    if 'quarter_label' in df.columns:
        df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if isinstance(x, str) and '-Q' in x else 0)
        drop_cols.append('quarter_label')
    
    target_col = 'construction_cost_per_m2_usd'
    
    # Drop columns
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop + [target_col])
    
    # Log Transform for RMSLE
    y = np.log1p(df[target_col])
    
    # Categoricals
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')

    # Split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training shape: {X_train.shape}, Validation shape: {X_valid.shape}")
    print(f"Features: {X_train.columns.tolist()}")

    # 4. Training (Optimized Params)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': device,
        'gpu_platform_id': 0, 
        'gpu_device_id': 0,
        'learning_rate': 0.03,
        'num_leaves': 31, # Can increase if underfitting, but 31 is safe
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 
        'bagging_freq': 5,
        'min_child_samples': 20, # Reduced slightly since we have more features now
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'colsample_bytree': 0.7, # Lower column sample to handle high dimensionality from image stats
        'max_depth': -1,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=cat_cols)
    
    print("Starting training Method 03...")
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=1500, # Increased rounds
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=50)]
    )
    
    # Check RMSLE
    val_preds = np.expm1(bst.predict(X_valid))
    val_actual = np.expm1(y_valid)
    val_rmsle = np.sqrt(mean_squared_error(np.log1p(val_actual), np.log1p(val_preds)))
    print(f"Method 03 Validation RMSLE: {val_rmsle}")
    
    bst.save_model('model_checkpoint_03.txt')
    print("Model saved to model_checkpoint_03.txt")

if __name__ == "__main__":
    train(None)
