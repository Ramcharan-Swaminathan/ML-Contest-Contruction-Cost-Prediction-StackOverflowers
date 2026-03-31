import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import os
import argparse
import rasterio
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Use CPU for extraction if CUDA not available for Torch
# (LightGBM uses CUDA separately)
EXTRACTION_DEVICE = torch.device('cpu') 

def get_resnet_embedder():
    """
    Returns a pretrained ResNet18 model with the last layer removed.
    """
    print("Loading pretrained ResNet18...")
    resnet = models.resnet18(pretrained=True)
    # Remove the last fully connected layer
    modules = list(resnet.children())[:-1]
    embedder = torch.nn.Sequential(*modules)
    embedder.to(EXTRACTION_DEVICE)
    embedder.eval()
    return embedder

def extract_cnn_embeddings(df, image_dir):
    """
    Extracts 512-dim embeddings from Sentinel-2 RGB images.
    """
    print("Extracting CNN embeddings (this will take time)...")
    
    embedder = get_resnet_embedder()
    
    # Preprocessing for ResNet (ImageNet stats)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    s2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    # RGB = B4 (Red), B3 (Green), B2 (Blue)
    
    embeddings = []
    valid_indices = []
    
    total = len(df)
    
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"Processing row {idx}/{total}")
            
        s2_filename = row.get('sentinel2_tiff_file_name')
        s2_path = os.path.join(image_dir, s2_filename) if isinstance(s2_filename, str) else None
        
        emb_vector = None
        
        if s2_path and os.path.exists(s2_path):
            try:
                with rasterio.open(s2_path) as src:
                    # Read only RGB (B4, B3, B2 are indices 3, 2, 1)
                    # Note: rasterio 1-based indexing in `read(i)`, but `read()` gives 0-based array
                    # Bands list 0-based: B1=0, B2=1, B3=2, B4=3
                    r = src.read(4) # Red
                    g = src.read(3) # Green
                    b = src.read(2) # Blue
                    
                    # Normalize to 0-255 for PIL (assuming 16-bit ints roughly 0-3000 or similar)
                    # Simple min-max or percentile scaling is robust
                    def normalize(band):
                        band = band.astype(float)
                        # clip to 98% percentile to remove outliers
                        p98 = np.percentile(band, 98)
                        if p98 > 0:
                            band = np.clip(band, 0, p98) / p98
                        else:
                            # Avoid div/0
                            band = np.clip(band, 0, 1) # unlikely
                        return (band * 255).astype(np.uint8)
                    
                    rgb = np.dstack((normalize(r), normalize(g), normalize(b)))
                    
                    pil_img = Image.fromarray(rgb)
                    input_tensor = preprocess(pil_img)
                    input_batch = input_tensor.unsqueeze(0).to(EXTRACTION_DEVICE)
                    
                    with torch.no_grad():
                        output = embedder(input_batch)
                        # ResNet18 output shape: [1, 512, 1, 1] -> flatten to [512]
                        emb_vector = output.cpu().numpy().flatten()
                        
            except Exception as e:
                # print(f"Error: {e}")
                pass
        
        if emb_vector is None:
            # Fallback: Zero vector (not ideal but safe)
            emb_vector = np.zeros(512)
            
        embeddings.append(emb_vector)
        
    embeddings = np.array(embeddings)
    
    # PCA to reduce dimensionality (512 is too much for 800 samples)
    # We'll reduce to 16 components
    n_components = 16
    print(f"Reducing embeddings from 512 to {n_components} using PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    # Fit PCA and transform
    # Note: In production we should fit on train and transform test.
    # Here we are inside 'train.py', so we can fit.
    # The 'predict.py' will need this fitted PCA.
    # To simplify, we will just save the PCA object or re-fit if we assume similar distribution.
    # BETTER: Save PCA model.
    import pickle
    
    pca_embeddings = pca.fit_transform(embeddings)
    
    with open('pca_model_05.pkl', 'wb') as f:
        pickle.dump(pca, f)
        
    # Add to DF
    emb_cols = [f'emb_{i}' for i in range(n_components)]
    emb_df = pd.DataFrame(pca_embeddings, columns=emb_cols, index=df.index)
    
    return pd.concat([df, emb_df], axis=1), emb_cols

def train(args):
    np.random.seed(42)
    device = 'cuda'
    
    data_path = 'dataset/train_tabular.csv'
    image_dir = 'dataset/train_composite'
    
    df = pd.read_csv(data_path)
    
    # 1. Extract Embeddings (Method 05)
    df, emb_cols = extract_cnn_embeddings(df, image_dir)
    
    # 2. Preprocess 
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name']
    if 'quarter_label' in df.columns:
        df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if isinstance(x, str) and '-Q' in x else 0)
        drop_cols.append('quarter_label')
        
    target_col = 'construction_cost_per_m2_usd'
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop + [target_col])
    
    y = np.log1p(df[target_col])
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
        
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Features: {X_train.columns.tolist()}")
    
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
        'colsample_bytree': 0.7,
        'max_depth': -1,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=cat_cols)
    
    print("Starting training Method 05...")
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
    print(f"Method 05 Validation RMSLE: {val_rmsle}")
    
    bst.save_model('model_checkpoint_05.txt')
    print("Model saved to model_checkpoint_05.txt") # PCA is saved in extract_cnn_embeddings

if __name__ == "__main__":
    train(None)
