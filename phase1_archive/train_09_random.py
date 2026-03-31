import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
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
import random
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
                            rotated_embeddings.append(output.cpu().numpy().flatten())
                            
                    emb_vector = np.mean(rotated_embeddings, axis=0)
                        
            except Exception:
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
    with open('pca_model_09.pkl', 'wb') as f:
        pickle.dump(pca, f)
        
    emb_cols = [f'emb_{i}' for i in range(n_components)]
    emb_df = pd.DataFrame(pca_embeddings, columns=emb_cols, index=df.index)
    
    return pd.concat([df, emb_df], axis=1)

def train_random_search(args):
    np.random.seed(42)
    random.seed(42)
    
    data_path = 'dataset/train_tabular.csv'
    image_dir = 'dataset/train_composite'
    
    df = pd.read_csv(data_path)
    
    # 1. Reuse/Extract Features with TTA
    df_processed = extract_embeddings_tta(df, image_dir)
    
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name']
    
    if 'quarter_label' in df_processed.columns:
        df_processed['quarter'] = df_processed['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if isinstance(x, str) and '-Q' in x else 0)
        drop_cols.append('quarter_label')
        
    target_col = 'construction_cost_per_m2_usd'
    cols_to_drop = [c for c in drop_cols if c in df_processed.columns]
    
    X = df_processed.drop(columns=cols_to_drop + [target_col])
    y = np.log1p(df_processed[target_col])
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
        
    print(f"Features ready. Starting Random Search (25 trials)...")
    
    best_rmse = float('inf')
    best_params = {}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
    for i in range(25):
        # Sample Params
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': random.uniform(0.005, 0.1),
            'num_leaves': random.randint(20, 100),
            'feature_fraction': random.uniform(0.4, 1.0),
            'bagging_fraction': random.uniform(0.4, 1.0),
            'bagging_freq': random.randint(1, 7),
            'min_child_samples': random.randint(5, 100),
            'reg_alpha': 10 ** random.uniform(-8, 1),
            'reg_lambda': 10 ** random.uniform(-8, 1),
            'colsample_bytree': random.uniform(0.4, 1.0)
        }
        
        rmses = []
        for train_idx, valid_idx in kf.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, categorical_feature=cat_cols)
            
            bst = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dvalid], 
                            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            
            preds = bst.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            rmses.append(rmse)
        
        avg_rmse = np.mean(rmses)
        print(f"Trial {i+1}/25 - RMSE: {avg_rmse:.5f} - Params: LR={params['learning_rate']:.3f}, Leaves={params['num_leaves']}")
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params
            print(f"  --> New Best RMSE: {best_rmse}")
            
    print('Best RMSE:', best_rmse)
    print('Best Params:', best_params)
    
    # Train Data with Best Params
    # We will train on 80/20 using best params for the checkpoint.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=cat_cols)
    
    print("Training Final Model with Best Params...")
    bst = lgb.train(
        best_params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=50)]
    )
    
    bst.save_model('model_checkpoint_09.txt')
    print("Best model saved to model_checkpoint_09.txt")

if __name__ == "__main__":
    train_random_search(None)
