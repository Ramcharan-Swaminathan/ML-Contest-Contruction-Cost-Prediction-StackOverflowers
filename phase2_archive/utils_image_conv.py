import os
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import concurrent.futures

# --- Config ---
TRAIN_IMG_DIR = 'dataset/train_composite'
TEST_IMG_DIR = 'evaluation_dataset/evaluation_composite'
TRAIN_OUT_DIR = 'dataset/train_png'
TEST_OUT_DIR = 'evaluation_dataset/test_png'

os.makedirs(TRAIN_OUT_DIR, exist_ok=True)
os.makedirs(TEST_OUT_DIR, exist_ok=True)

def convert_single_image(args):
    filename, src_dir, dst_dir = args
    if not isinstance(filename, str) or not filename.endswith('.tif'):
        return None
    
    src_path = os.path.join(src_dir, filename)
    dst_filename = filename.replace('.tif', '.png')
    dst_path = os.path.join(dst_dir, dst_filename)
    
    # Skip if exists? No, overwrite to be safe.
    # if os.path.exists(dst_path): return dst_filename
    
    try:
        with rasterio.open(src_path) as src:
            # Assuming Sentinel-2: R=4, G=3, B=2
            # Handle potential index errors if bands are missing
            try:
                r = src.read(4)
                g = src.read(3)
                b = src.read(2)
            except IndexError:
                # Fallback for VIIRS or weird files (treat as grayscale or just 1,2,3)
                r = src.read(1)
                g = src.read(1)
                b = src.read(1)
                
            def norm(x):
                x = np.nan_to_num(x).astype(float)
                # Robust scaling using 98th percentile to avoid outliers
                p98 = np.percentile(x, 98)
                if p98 > 0:
                    x = np.clip(x, 0, p98) / p98
                else:
                    x = np.clip(x, 0, 1) # Fallback
                return (x * 255).astype(np.uint8)
            
            rgb = np.dstack((norm(r), norm(g), norm(b)))
            img = Image.fromarray(rgb)
            img.save(dst_path)
            return dst_filename
    except Exception as e:
        print(f"Failed to convert {filename}: {e}")
        return None

def process_directory(csv_path, src_dir, dst_dir, col_name='sentinel2_tiff_file_name'):
    print(f"Processing {src_dir} -> {dst_dir}...")
    df = pd.read_csv(csv_path)
    files = df[col_name].dropna().unique().tolist()
    
    tasks = [(f, src_dir, dst_dir) for f in files]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(convert_single_image, tasks), total=len(tasks)))
    
    print(f"Converted {len([r for r in results if r])} images.")

if __name__ == "__main__":
    process_directory('dataset/train_tabular.csv', TRAIN_IMG_DIR, TRAIN_OUT_DIR)
    process_directory('evaluation_tabular_no_target.csv', TEST_IMG_DIR, TEST_OUT_DIR)
