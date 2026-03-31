import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import argparse
import rasterio
import warnings

warnings.filterwarnings('ignore')

def calculate_indices(s2_data, s2_bands):
    """
    Calculates spectral indices from Sentinel-2 data.
    MUST MATCH train_04.py LOGIC EXACTLY.
    """
    def get_band(name):
        try:
            return s2_data[s2_bands.index(name)].astype(float)
        except:
            return None

    B3 = get_band('B3')
    B4 = get_band('B4')
    B8 = get_band('B8')
    B11 = get_band('B11')

    indices = {}
    
    # 1. NDVI (Vegetation)
    if B8 is not None and B4 is not None:
        denom = (B8 + B4)
        denom[denom == 0] = np.nan 
        indices['NDVI'] = (B8 - B4) / denom

    # 2. NDBI (Built-up)
    if B11 is not None and B8 is not None:
        denom = (B11 + B8)
        denom[denom == 0] = np.nan
        indices['NDBI'] = (B11 - B8) / denom
        
    # 3. NDWI (Water)
    if B3 is not None and B8 is not None:
        denom = (B3 + B8)
        denom[denom == 0] = np.nan
        indices['NDWI'] = (B3 - B8) / denom

    return indices

def extract_advanced_features(df, image_dir):
    print("Extracting ADVANCED image features for prediction...")
    
    s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    features_list = []
    total = len(df)
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{total}")
            
        s2_filename = row.get('sentinel2_tiff_file_name')
        s2_path = os.path.join(image_dir, s2_filename) if isinstance(s2_filename, str) else None
        
        row_feats = {}
        
        if s2_path and os.path.exists(s2_path):
            try:
                with rasterio.open(s2_path) as src:
                    img = src.read() 
                    indices_map = calculate_indices(img, s2_bands)
                    for name, data in indices_map.items():
                        row_feats[f'{name}_mean'] = np.nanmean(data)
                        row_feats[f'{name}_std'] = np.nanstd(data)
                        row_feats[f'{name}_max'] = np.nanmax(data)
                        
                    key_bands = ['B4', 'B8', 'B11']
                    for b in key_bands:
                         band_idx = s2_bands.index(b)
                         b_data = img[band_idx]
                         row_feats[f's2_{b}_mean'] = np.nanmean(b_data)
            except Exception:
                pass 
        
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

    feat_df = pd.DataFrame(features_list, index=df.index)
    return pd.concat([df, feat_df], axis=1)

def predict(args):
    # Args Handling
    if isinstance(args, argparse.Namespace):
        model_path = args.model_path 
        data_path = args.data_path
        output_path = args.output_path
        image_dir = args.image_dir
    else:
         model_path = 'model_checkpoint_04.txt'
         data_path = 'evaluation_tabular_no_target.csv'
         output_path = 'submission004.csv'
         image_dir = 'evaluation_dataset/evaluation_composite'

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
    ids = df['data_id'] if 'data_id' in df.columns else None

    # 1. Image Features
    df = extract_advanced_features(df, image_dir)
    
    # 2. Drop columns
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', 'construction_cost_per_m2_usd']
    
    if 'quarter_label' in df.columns:
        df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if isinstance(x, str) and '-Q' in x else 0)
    
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # 3. Categoricals
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
        
    # 4. Feature Matching
    model_features = bst.feature_name()
    missing = [f for f in model_features if f not in X.columns]
    
    # Fill missing features with NaN if any (for robustness) 
    # But print warning
    if missing:
        print(f"WARNING: Input data missing features: {missing}. Filling with NaN.")
        for f in missing:
            X[f] = np.nan
            
    X = X[model_features]
    print("Features matched.")
    
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
    parser.add_argument('--model_path', type=str, default='model_checkpoint_04.txt')
    parser.add_argument('--output_path', type=str, default='submission004.csv')
    parser.add_argument('--image_dir', type=str, default='evaluation_dataset/evaluation_composite')
    
    args = parser.parse_args()
    predict(args)
