import pandas as pd
import numpy as np
import rasterio
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
TRAIN_DATA = 'dataset/train_tabular.csv'
TRAIN_IMG_DIR = 'dataset/train_composite'
TEST_DATA = 'evaluation_tabular_no_target.csv'
TEST_IMG_DIR = 'evaluation_dataset/evaluation_composite'

OUTPUT_TRAIN = 'dataset/image_features_train.csv'
OUTPUT_TEST = 'evaluation_dataset/image_features_test.csv'

# S2 Bands: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 (Total 12)
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
STATS = ['mean', 'std', 'median', 'min', 'max']

def process_single_row(args):
    """
    Process a single row (data_id) to find and extract features from S2 and VIIRS images.
    """
    row, img_dir = args
    data_id = row['data_id']
    
    # Initialize result dictionary
    features = {'data_id': data_id}
    
    # --- Sentinel-2 ---
    s2_fname = row.get('sentinel2_tiff_file_name')
    if isinstance(s2_fname, str):
        path = os.path.join(img_dir, s2_fname)
        if os.path.exists(path):
            try:
                with rasterio.open(path) as src:
                    # Read all 12 bands
                    # rasterio reads (C, H, W)
                    data = src.read() 
                    
                    # Ensure we have 12 bands (or handled if fewer? User said 12)
                    # If count != 12, we iterate what we have or pad?
                    # Assuming 12 based on check.
                    
                    for b_idx in range(data.shape[0]):
                        band_name = S2_BANDS[b_idx] if b_idx < 12 else f"B_unk_{b_idx}"
                        band_data = data[b_idx].flatten()
                        
                        # Convert to float for stats and handle nodata if needed (usually 0 or specific header)
                        # We use np.nan* functions so let's ensure nodata is NaN if present, 
                        # but often raw values are valid. 
                        # If src.nodata is set, we use it.
                        
                        band_data = band_data.astype(np.float32)
                        
                        if src.nodata is not None:
                             band_data[band_data == src.nodata] = np.nan
                             
                        features[f's2_{band_name}_mean'] = np.nanmean(band_data)
                        features[f's2_{band_name}_std'] = np.nanstd(band_data)
                        features[f's2_{band_name}_median'] = np.nanmedian(band_data)
                        features[f's2_{band_name}_min'] = np.nanmin(band_data)
                        features[f's2_{band_name}_max'] = np.nanmax(band_data)
            except Exception as e:
                # print(f"Error reading S2 {path}: {e}")
                pass # Leaves features missing (NaN)
                
    # --- VIIRS ---
    viirs_fname = row.get('viirs_tiff_file_name')
    if isinstance(viirs_fname, str):
        path = os.path.join(img_dir, viirs_fname)
        if os.path.exists(path):
            try:
                with rasterio.open(path) as src:
                    # Read single band
                    data = src.read(1).flatten()
                    data = data.astype(np.float32)
                    
                    if src.nodata is not None:
                        data[data == src.nodata] = np.nan
                        
                    features['viirs_mean'] = np.nanmean(data)
                    features['viirs_std'] = np.nanstd(data)
                    features['viirs_median'] = np.nanmedian(data)
                    features['viirs_min'] = np.nanmin(data)
                    features['viirs_max'] = np.nanmax(data)
            except Exception as e:
                pass

    return features

def extract_dataset(csv_path, img_dir, output_path):
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Prepare args for parallel map
    # Convert rows to dicts for simpler pickling? or just iterate tuples
    # process_single_row takes (row_series, img_dir)
    
    # We turn df into a list of rows to avoid passing whole df to workers if poss, 
    # though with fork it's copy-on-write.
    
    tasks = [(row, img_dir) for _, row in df.iterrows()]
    
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_single_row, tasks), total=len(tasks)))
        
    # Convert to DataFrame
    feat_df = pd.DataFrame(results)
    
    # Merge with original data_id to ensure alignment/completeness
    # Actually results have data_id.
    
    print(f"Saving to {output_path}...")
    feat_df.to_csv(output_path, index=False)
    print("Done.")

def main():
    print("Starting Feature Extraction...")
    
    # Train
    if os.path.exists(TRAIN_DATA):
        extract_dataset(TRAIN_DATA, TRAIN_IMG_DIR, OUTPUT_TRAIN)
    else:
        print(f"Train data not found at {TRAIN_DATA}")
        
    # Test
    if os.path.exists(TEST_DATA):
        extract_dataset(TEST_DATA, TEST_IMG_DIR, OUTPUT_TEST)
    else:
        print(f"Test data not found at {TEST_DATA}")

if __name__ == "__main__":
    main()
