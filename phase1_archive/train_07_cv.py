import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
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

def extract_cnn_embeddings(df, image_dir):
    print("Extracting CNN embeddings (Method 07 - Shared)...")
    
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
                    input_tensor = preprocess(pil_img)
                    input_batch = input_tensor.unsqueeze(0).to(EXTRACTION_DEVICE)
                    
                    with torch.no_grad():
                        output = embedder(input_batch)
                        emb_vector = output.cpu().numpy().flatten()
                        
            except Exception:
                pass
        
        if emb_vector is None:
            emb_vector = np.zeros(512)
            
        embeddings.append(emb_vector)
        
    embeddings = np.array(embeddings)
    
    # PCA
    # Fit PCA on ALL data for stability in this script, or we can do it inside CV.
    # To match Method 05 success, we will fit globally here.
    n_components = 16
    print(f"Reducing embeddings from 512 to {n_components} using PCA (Global Fit)...")
    pca = PCA(n_components=n_components, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)
    
    # Save PCA model for inference
    with open('pca_model_07.pkl', 'wb') as f:
        pickle.dump(pca, f)
        
    emb_cols = [f'emb_{i}' for i in range(n_components)]
    emb_df = pd.DataFrame(pca_embeddings, columns=emb_cols, index=df.index)
    
    return pd.concat([df, emb_df], axis=1), emb_cols

def train_cv(args):
    np.random.seed(42)
    device = 'cuda'
    
    data_path = 'dataset/train_tabular.csv'
    image_dir = 'dataset/train_composite'
    
    df = pd.read_csv(data_path)
    
    # 1. Extract Embeddings
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
        
    print(f"Features: {X.columns.tolist()}")

    # 3. K-Fold CV
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(df))
    oof_targets = np.zeros(len(df))
    
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
        'colsample_bytree': 0.7, # Slightly lower colsample for CV variation
        'max_depth': -1,
        'verbose': -1
    }
    
    print(f"Starting {n_folds}-Fold Cross-Validation...")
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining Fold {fold+1}/{n_folds}...")
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=cat_cols)
        
        bst = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)] # Silence non-essential logs
        )
        
        # Save model for this fold
        model_name = f'model_cv_fold_{fold+1}.txt'
        bst.save_model(model_name)
        
        preds = bst.predict(X_valid)
        oof_preds[val_idx] = np.expm1(preds)
        oof_targets[val_idx] = np.expm1(y_valid)
        
        fold_rmse = np.sqrt(mean_squared_error(y_valid, preds)) # Metric on log scale = RMSLE
        scores.append(fold_rmse)
        print(f"Fold {fold+1} RMSLE: {fold_rmse:.5f}")
        
    print("\n" + "="*30)
    print(f"CV Scores: {scores}")
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Average CV RMSLE: {mean_score:.5f} +/- {std_score:.5f}")
    
    # Global OOF Score
    global_rmsle = np.sqrt(mean_squared_error(np.log1p(oof_targets), np.log1p(oof_preds)))
    print(f"Global OOF RMSLE: {global_rmsle:.5f}")
    print("="*30)

if __name__ == "__main__":
    train_cv(None)
