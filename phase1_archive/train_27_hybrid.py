import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import rasterio
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
TRAIN_DATA = 'dataset/train_tabular.csv'
TRAIN_IMG_DIR = 'dataset/train_composite'
TEST_DATA = 'evaluation_tabular_no_target.csv'
TEST_IMG_DIR = 'evaluation_dataset/evaluation_composite'
TRAIN_IMG_FEATS = 'dataset/image_features_train.csv'
TEST_IMG_FEATS = 'evaluation_dataset/image_features_test.csv'
PSEUDO_LABEL_FILE = 'submission_final.csv' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Hyperparameters (Regularized for High Dim) ---
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.015, # Slightly lower
    'num_leaves': 60,       # Slightly less complex trees
    'feature_fraction': 0.7, # Sample fewer features per tree to handle noise
    'bagging_fraction': 0.90,
    'bagging_freq': 5,
    'min_child_samples': 30, # Higher to prevent overfitting
    'reg_alpha': 0.5,       # Stronger L1
    'reg_lambda': 0.5,      # Stronger L2
    'colsample_bytree': 0.7
}

# --- 1. ResNet Embedding Logic (M21) ---
def get_resnet_embedder():
    print("Loading ResNet18...")
    resnet = models.resnet18(pretrained=True)
    modules = list(resnet.children())[:-1]
    embedder = torch.nn.Sequential(*modules).to(DEVICE)
    embedder.eval()
    return embedder

def extract_embeddings(df, image_dir, embedder):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    embeddings = []
    # Simple TTA (Center Crop vs Resize? Just Resize)
    # Using 4x TTA as M21 did
    
    print(f"Extracting embeddings from {image_dir}...")
    # Batch processing would be faster but manual loop is fine for 1000 images
    
    for idx, row in df.iterrows():
        s2_filename = row.get('sentinel2_tiff_file_name')
        s2_path = os.path.join(image_dir, s2_filename) if isinstance(s2_filename, str) else None
        
        emb_vector = None
        if s2_path and os.path.exists(s2_path):
            try:
                with rasterio.open(s2_path) as src:
                    r, g, b = src.read(4), src.read(3), src.read(2)
                    def norm(x):
                        x = np.nan_to_num(x).astype(float)
                        p98 = np.percentile(x, 98)
                        if p98 > 0: x = np.clip(x, 0, p98)/p98
                        else: x = np.clip(x, 0, 1)
                        return (x*255).astype(np.uint8)
                    rgb = np.dstack((norm(r), norm(g), norm(b)))
                    img = Image.fromarray(rgb)
                    
                    # TTA 4x
                    rot_embs = []
                    for angle in [0, 90, 180, 270]:
                        t_img = img.rotate(angle) if angle > 0 else img
                        t_tensor = preprocess(t_img).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            rot_embs.append(embedder(t_tensor).cpu().numpy().flatten())
                    emb_vector = np.mean(rot_embs, axis=0)
            except: pass
            
        if emb_vector is None:
            emb_vector = np.zeros(512)
        embeddings.append(emb_vector)
        
    return np.array(embeddings)

# --- 2. Explicit Stats & Research Features (M24) ---
def load_extracted_feats(df, path):
    if os.path.exists(path):
        feats = pd.read_csv(path)
        df = df.merge(feats, on='data_id', how='left')
    return df

def feature_engineering(df):
    # Research Features
    if 'straight_distance_to_capital_km' in df.columns:
        df['is_remote_island'] = (df['straight_distance_to_capital_km'] > 800).astype(int)
        
    if 'flood_risk' in df.columns and 'seismic_risk' in df.columns:
        fr = df['flood_risk'].fillna(df['flood_risk'].median())
        sr = df['seismic_risk'].fillna(df['seismic_risk'].median())
        df['risk_index'] = fr + sr

    if 'quarter_label' in df.columns:
        df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if '-Q' in str(x) else 0)
        
    return df

# --- 3. Meta Features (M20/M21) ---
def add_spatial_encoding(train_df, test_df, pseudo_df=None):
    # Method 20 Logic
    print("Adding Spatial Target Encodings...")
    target_cols = ['geolocation_name', 'country', 'region_economic_classification']
    train_df['log_target'] = np.log1p(train_df['construction_cost_per_m2_usd'])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for col in target_cols:
        train_df[f'{col}_target_enc'] = 0.0
        for tr_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            global_mean = X_tr['log_target'].mean()
            mapping = X_tr.groupby(col)['log_target'].mean()
            train_df.loc[val_idx, f'{col}_target_enc'] = X_val[col].map(mapping).fillna(global_mean)
    mappings = {}
    global_mean = train_df['log_target'].mean()
    for col in target_cols:
        mappings[col] = train_df.groupby(col)['log_target'].mean()
        test_df[f'{col}_target_enc'] = test_df[col].map(mappings[col]).fillna(global_mean)
        if pseudo_df is not None:
             pseudo_df[f'{col}_target_enc'] = pseudo_df[col].map(mappings[col]).fillna(global_mean)
    train_df = train_df.drop(columns=['log_target'])
    return train_df, test_df, pseudo_df

def add_visual_knn_feature(train_df, test_df, pseudo_df, train_embs, test_embs, k=50):
    print(f"Adding Visual KNN Feature (K={k})...")
    
    y_train = np.log1p(train_df['construction_cost_per_m2_usd'].values)
    
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn.fit(train_embs)
    
    # Train Feature (OOF)
    dists, indices = nn.kneighbors(train_embs, n_neighbors=k+1)
    train_knn = []
    for i in range(len(train_embs)):
        neighbor_idxs = indices[i, 1:] # Skip self
        mean_val = np.mean(y_train[neighbor_idxs])
        train_knn.append(mean_val)
    train_df['visual_knn_cost'] = train_knn
    
    # Test Feature
    dists, indices = nn.kneighbors(test_embs, n_neighbors=k)
    test_knn = []
    for i in range(len(test_embs)):
        neighbor_idxs = indices[i]
        mean_val = np.mean(y_train[neighbor_idxs])
        test_knn.append(mean_val)
    test_df['visual_knn_cost'] = test_knn
    pseudo_df['visual_knn_cost'] = test_knn
    
    return train_df, test_df, pseudo_df

def train_hybrid():
    print("Loading Data...")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    pseudo_df = pd.read_csv(PSEUDO_LABEL_FILE)
    
    # Pseudo Prep
    if 'data_id' in test_df.columns and 'data_id' in pseudo_df.columns:
        pseudo_df_full = test_df.copy()
        label_map = dict(zip(pseudo_df['data_id'], pseudo_df['construction_cost_per_m2_usd']))
        pseudo_df_full['construction_cost_per_m2_usd'] = pseudo_df_full['data_id'].map(label_map)
    else:
        pseudo_df_full = test_df.copy()
        pseudo_df_full['construction_cost_per_m2_usd'] = pseudo_df['construction_cost_per_m2_usd']
        
    # --- Step 1: Explicit Features (Stats + Research) ---
    print("Step 1: Explicit Features...")
    train_df = load_extracted_feats(train_df, TRAIN_IMG_FEATS)
    test_df = load_extracted_feats(test_df, TEST_IMG_FEATS)
    pseudo_df_full = load_extracted_feats(pseudo_df_full, TEST_IMG_FEATS) # Pseudo uses test feats
    
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    pseudo_df_full = feature_engineering(pseudo_df_full)
    
    train_df, test_df, pseudo_df_full = add_spatial_encoding(train_df, test_df, pseudo_df_full)

    # --- Step 2: Deep Embeddings ---
    print("Step 2: Deep Learning Embeddings...")
    embedder = get_resnet_embedder()
    train_embs = extract_embeddings(train_df, TRAIN_IMG_DIR, embedder)
    test_embs = extract_embeddings(test_df, TEST_IMG_DIR, embedder)
    
    # Add embeddings to dataframe
    emb_cols = [f'emb_{i}' for i in range(512)]
    train_emb_df = pd.DataFrame(train_embs, columns=emb_cols, index=train_df.index)
    test_emb_df = pd.DataFrame(test_embs, columns=emb_cols, index=test_df.index)
    
    # Visual KNN (using raw embeddings)
    train_df, test_df, pseudo_df_full = add_visual_knn_feature(
        train_df, test_df, pseudo_df_full, 
        train_embs, test_embs, k=50
    )
    
    # Merge Embeddings
    train_df = pd.concat([train_df, train_emb_df], axis=1)
    test_df = pd.concat([test_df, test_emb_df], axis=1)
    pseudo_df_full = pd.concat([pseudo_df_full, test_emb_df], axis=1) # Pseudo uses test embeddings
    
    # --- Step 3: Training ---
    print("Step 3: Training...")
    
    # Prep Categoricals
    for df in [train_df, test_df, pseudo_df_full]:
         for col in df.select_dtypes(include=['object']).columns:
            if col not in ['sentinel2_tiff_file_name', 'viirs_tiff_file_name']:
                df[col] = df[col].astype('category')

    target_col = 'construction_cost_per_m2_usd'
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', target_col]
    features = [c for c in train_df.columns if c not in drop_cols]
    
    print(f"Total Features: {len(features)}")
    
    X = train_df[features]
    y = np.log1p(train_df[target_col])
    groups = train_df['geolocation_name'] 
    
    X_pseudo = pseudo_df_full[features]
    y_pseudo = np.log1p(pseudo_df_full[target_col])
    
    # GroupKFold Validation
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(X))
    
    print("Starting GroupKFold Cross-Validation...")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Add Pseudo Data to Training Fold
        X_tr_fold = pd.concat([X_tr, X_pseudo], axis=0)
        y_tr_fold = pd.concat([y_tr, y_pseudo], axis=0)
        
        train_data = lgb.Dataset(X_tr_fold, label=y_tr_fold)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        bst = lgb.train(params, train_data, num_boost_round=5000, 
                        valid_sets=[val_data], 
                        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)])
                        
        oof_preds[val_idx] = bst.predict(X_val)
        print(f"Fold {fold+1} RMSLE: {np.sqrt(mean_squared_error(y_val, oof_preds[val_idx])):.5f}")
        
    score = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"GroupKFold Global RMSLE Score: {score:.5f}")
    
    # Retrain Full
    print("Retraining on Full Data...")
    X_full = pd.concat([X, X_pseudo], axis=0)
    y_full = pd.concat([y, y_pseudo], axis=0)
    
    train_data = lgb.Dataset(X_full, label=y_full)
    bst = lgb.train(params, train_data, num_boost_round=1500)
    
    preds_log = bst.predict(test_df[features])
    preds = np.expm1(preds_log)
    
    sub = pd.DataFrame()
    if 'data_id' in test_df.columns:
        sub['data_id'] = test_df['data_id']
    sub['construction_cost_per_m2_usd'] = preds
    sub.to_csv('submission027.csv', index=False)
    print("Done. Saved submission027.csv")

if __name__ == "__main__":
    train_hybrid()
