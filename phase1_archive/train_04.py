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

def calculate_indices(s2_data, s2_bands):
    """
    Calculates spectral indices from Sentinel-2 data.
    s2_data: (12, H, W) numpy array
    s2_bands: list of band names corresponding to indices 0-11
    """
    # Map band names to indices
    # Bands: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
    # Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    
    # Helper to get band by name safely
    def get_band(name):
        try:
            return s2_data[s2_bands.index(name)].astype(float)
        except:
            return None

    B3 = get_band('B3') # Green
    B4 = get_band('B4') # Red
    B8 = get_band('B8') # NIR
    B11 = get_band('B11') # SWIR

    indices = {}

    # 1. NDVI (Normalized Difference Vegetation Index)
    # (NIR - Red) / (NIR + Red)
    if B8 is not None and B4 is not None:
        denom = (B8 + B4)
        # Avoid division by zero
        denom[denom == 0] = np.nan 
        indices['NDVI'] = (B8 - B4) / denom

    # 2. NDBI (Normalized Difference Built-up Index)
    # (SWIR - NIR) / (SWIR + NIR) -> Critical for construction!
    if B11 is not None and B8 is not None:
        denom = (B11 + B8)
        denom[denom == 0] = np.nan
        indices['NDBI'] = (B11 - B8) / denom
        
    # 3. NDWI (Normalized Difference Water Index)
    # (Green - NIR) / (Green + NIR)
    if B3 is not None and B8 is not None:
        denom = (B3 + B8)
        denom[denom == 0] = np.nan
        indices['NDWI'] = (B3 - B8) / denom

    return indices

def extract_advanced_features(df, image_dir):
    """
    Extracts advanced spectral indices and basic stats.
    """
    print("Extracting ADVANCED image features (Indices)...")
    
    s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    
    features_list = []
    
    total = len(df)
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{total}")
            
        # --- Sentinel-2 ---
        s2_filename = row.get('sentinel2_tiff_file_name')
        s2_path = os.path.join(image_dir, s2_filename) if isinstance(s2_filename, str) else None
        
        row_feats = {}
        
        if s2_path and os.path.exists(s2_path):
            try:
                with rasterio.open(s2_path) as src:
                    img = src.read() 
                    
                    # 1. Calculate Indices
                    indices_map = calculate_indices(img, s2_bands)
                    
                    for name, data in indices_map.items():
                        row_feats[f'{name}_mean'] = np.nanmean(data)
                        row_feats[f'{name}_std'] = np.nanstd(data)
                        row_feats[f'{name}_max'] = np.nanmax(data)
                        
                    # 2. Keep Basic Stats for key bands only (Red, NIR, SWIR involved in indices)
                    # To reduce dimensionality noise compared to Method 03
                    key_bands = ['B4', 'B8', 'B11']
                    for b in key_bands:
                         band_idx = s2_bands.index(b)
                         b_data = img[band_idx]
                         row_feats[f's2_{b}_mean'] = np.nanmean(b_data)
                    
            except Exception as e:
                # print(f"Error reading Sentinel-2 {s2_path}: {e}")
                pass # Defaults to NaN
        
        # --- VIIRS ---
        # Keep VIIRS as it's a strong feature for development
        viirs_filename = row.get('viirs_tiff_file_name')
        viirs_path = os.path.join(image_dir, viirs_filename) if isinstance(viirs_filename, str) else None
        
        if viirs_path and os.path.exists(viirs_path):
            try:
                with rasterio.open(viirs_path) as src:
                    img = src.read(1)
                    row_feats['viirs_mean'] = np.nanmean(img)
                    row_feats['viirs_max'] = np.nanmax(img)
            except Exception:
                 pass

        features_list.append(row_feats)

    # DataFrame construction
    feat_df = pd.DataFrame(features_list, index=df.index)
    
    return pd.concat([df, feat_df], axis=1)

def train(args):
    np.random.seed(42)
    device = 'cuda'
    
    # 1. Load Data
    data_path = 'dataset/train_tabular.csv'
    image_dir = 'dataset/train_composite'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")

    df = pd.read_csv(data_path)
    
    # 2. Extract Image Features (Method 04)
    df = extract_advanced_features(df, image_dir)
    
    # 3. Preprocessing 
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name']
    
    if 'quarter_label' in df.columns:
        df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if isinstance(x, str) and '-Q' in x else 0)
        drop_cols.append('quarter_label')
    
    target_col = 'construction_cost_per_m2_usd'
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop + [target_col])
    
    # Log Transform Target
    y = np.log1p(df[target_col])
    
    # Categoricals
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')

    # Split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Features: {X_train.columns.tolist()}")

    # 4. Training
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': device,
        'gpu_platform_id': 0, 
        'gpu_device_id': 0,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 
        'bagging_freq': 5,
        'min_child_samples': 20, 
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'colsample_bytree': 0.8,
        'max_depth': -1,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=cat_cols)
    
    print("Starting training Method 04...")
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=50)]
    )
    
    val_preds = np.expm1(bst.predict(X_valid))
    val_actual = np.expm1(y_valid)
    val_rmsle = np.sqrt(mean_squared_error(np.log1p(val_actual), np.log1p(val_preds)))
    print(f"Method 04 Validation RMSLE: {val_rmsle}")
    
    bst.save_model('model_checkpoint_04.txt')
    print("Model saved to model_checkpoint_04.txt")

if __name__ == "__main__":
    train(None)
