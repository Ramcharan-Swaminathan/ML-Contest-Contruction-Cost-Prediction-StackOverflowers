import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import os
import argparse
import rasterio
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import warnings

warnings.filterwarnings('ignore')

EXTRACTION_DEVICE = torch.device('cpu') 

def get_resnet_embedder():
    print("Loading pretrained ResNet18...")
    resnet = models.resnet18(pretrained=True)
    modules = list(resnet.children())[:-1]
    embedder = torch.nn.Sequential(*modules)
    embedder.to(EXTRACTION_DEVICE)
    embedder.eval()
    return embedder

def extract_embeddings_tta(df, image_dir):
    print("Extracting CNN embeddings with TTA (4x Rotations)...")
    
    embedder = get_resnet_embedder()
    
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    embeddings = []
    
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
                    r = src.read(4) 
                    g = src.read(3) 
                    b = src.read(2) 
                    
                    def normalize(band):
                        band = band.astype(float)
                        p98 = np.percentile(band, 98)
                        if p98 > 0:
                            band = np.clip(band, 0, p98) / p98
                        else:
                            band = np.clip(band, 0, 1)
                        return (band * 255).astype(np.uint8)
                    
                    rgb = np.dstack((normalize(r), normalize(g), normalize(b)))
                    pil_img = Image.fromarray(rgb)
                    
                    # TTA: 0, 90, 180, 270 degrees
                    rotations = [0, 90, 180, 270]
                    rotated_embeddings = []
                    
                    for angle in rotations:
                        if angle == 0:
                            img_t = pil_img
                        else:
                            img_t = pil_img.rotate(angle)
                            
                        input_tensor = preprocess(img_t)
                        input_batch = input_tensor.unsqueeze(0).to(EXTRACTION_DEVICE)
                        
                        with torch.no_grad():
                            output = embedder(input_batch)
                            # Flatten 512
                            rotated_embeddings.append(output.cpu().numpy().flatten())
                            
                    # Average the 4 vectors
                    emb_vector = np.mean(rotated_embeddings, axis=0)
                        
            except Exception as e:
                # print(e)
                pass
        
        if emb_vector is None:
            emb_vector = np.zeros(512)
            
        embeddings.append(emb_vector)
        
    embeddings = np.array(embeddings)
    
    # PCA
    n_components = 16
    print(f"Reducing embeddings from 512 to {n_components} using PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)
    
    # Save PCA model for inference
    with open('pca_model_08.pkl', 'wb') as f:
        pickle.dump(pca, f)
        
    emb_cols = [f'emb_{i}' for i in range(n_components)]
    emb_df = pd.DataFrame(pca_embeddings, columns=emb_cols, index=df.index)
    
    return pd.concat([df, emb_df], axis=1)

def train_tta(args):
    np.random.seed(42)
    device = 'cuda' # Assuming cuda is available
    
    data_path = 'dataset/train_tabular.csv'
    image_dir = 'dataset/train_composite'
    
    df = pd.read_csv(data_path)
    
    # 1. Extract Features with TTA
    df = extract_embeddings_tta(df, image_dir)
    
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
        
    print(f"Features: {X.columns.tolist()}")
    
    # 3. Split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=cat_cols)
    
    # 4. Train
    print("Starting training Method 08 (TTA)...")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': device,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'feature_fraction': 0.8, 
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 30, # slightly higher for stability
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'colsample_bytree': 0.8,
        'verbose': -1
    }
    
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=50)]
    )
    
    val_preds = bst.predict(X_valid)
    val_rmsle = np.sqrt(mean_squared_error(y_valid, val_preds))
    print(f"Method 08 (TTA) Validation RMSLE: {val_rmsle}")
    
    bst.save_model('model_checkpoint_08.txt')
    print("Model saved to model_checkpoint_08.txt")

if __name__ == "__main__":
    train_tta(None)
