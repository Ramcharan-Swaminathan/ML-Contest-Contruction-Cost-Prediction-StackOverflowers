import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import argparse
import rasterio
import warnings

warnings.filterwarnings('ignore')

def extract_image_features(df, image_dir):
    """
    Extracts statistical features from Sentinel-2 and VIIRS images.
    MUST MATCH train_03.py LOGIC EXACTLY.
    """
    print("Extracting image features for prediction...")
    
    s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    s2_features = []
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
                    img = src.read() 
                    for i, band_name in enumerate(s2_bands):
                        band_data = img[i, :, :]
                        s2_stats[f's2_{band_name}_mean'] = np.nanmean(band_data)
                        s2_stats[f's2_{band_name}_std'] = np.nanstd(band_data)
                        s2_stats[f's2_{band_name}_max'] = np.nanmax(band_data)
            except Exception as e:
                # print(f"Error reading Sentinel-2 {s2_path}: {e}")
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
                    img = src.read(1)
                    viirs_stats['viirs_mean'] = np.nanmean(img)
                    viirs_stats['viirs_max'] = np.nanmax(img)
                    viirs_stats['viirs_std'] = np.nanstd(img)
            except Exception:
                 viirs_stats['viirs_mean'] = np.nan
                 viirs_stats['viirs_max'] = np.nan
                 viirs_stats['viirs_std'] = np.nan
        else:
             viirs_stats['viirs_mean'] = np.nan
             viirs_stats['viirs_max'] = np.nan
             viirs_stats['viirs_std'] = np.nan
        
        viirs_features.append(viirs_stats)

    s2_df = pd.DataFrame(s2_features, index=df.index)
    viirs_df = pd.DataFrame(viirs_features, index=df.index)
    
    return pd.concat([df, s2_df, viirs_df], axis=1)

def predict(args):
    # Support direct call or argparse
    if isinstance(args, argparse.Namespace):
        model_path = args.model_path 
        data_path = args.data_path
        output_path = args.output_path
        image_dir = args.image_dir
    else:
         # Defaults
         model_path = 'model_checkpoint_03.txt'
         data_path = 'dataset/train_tabular.csv' # Fallback default
         output_path = 'submission003.csv'
         image_dir = 'dataset/train_composite'

    if data_path is None:
        data_path = 'dataset/train_tabular.csv'
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data not found at {data_path}")

    print(f"Loading model from {model_path}...")
    bst = lgb.Booster(model_file=model_path)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Store IDs
    ids = df['data_id'] if 'data_id' in df.columns else None

    # --- Preprocessing ---
    # 1. Image Features
    df = extract_image_features(df, image_dir)
    
    # 2. Drop columns
    # Note: geolocation_name dropped in train_03.py? Warning check.
    # Actually train_03.py drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name']
    # And quarters appended.
    
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', 'construction_cost_per_m2_usd']
    
    # Extract Quarter
    if 'quarter_label' in df.columns:
        df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if isinstance(x, str) and '-Q' in x else 0)
    
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # 3. Categorical Columns
    # Dynamic detection + explicit cast to match train
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
        
    # 4. Feature Matching
    model_features = bst.feature_name()
    
    # Handle missing features (e.g. if some image feature is missing in test df logic?)
    # They should be there because we just ran extract_image_features.
    
    # Check for missing
    missing = [f for f in model_features if f not in X.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
        
    # Reorder
    X = X[model_features]
    print("Features matched and reordered.")
    
    # Predict
    preds_log = bst.predict(X)
    preds_actual = np.expm1(preds_log)
    
    submission = pd.DataFrame()
    if ids is not None:
        submission['data_id'] = ids
        
    submission['construction_cost_per_m2_usd'] = preds_actual
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str, default='model_checkpoint_03.txt')
    parser.add_argument('--output_path', type=str, default='submission003.csv')
    parser.add_argument('--image_dir', type=str, default='dataset/train_composite')
    
    args = parser.parse_args()
    predict(args)
