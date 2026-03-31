import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import rasterio
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
TRAIN_DATA = 'dataset/train_tabular.csv'
TRAIN_IMG_DIR = 'dataset/train_composite'
TEST_DATA = 'evaluation_tabular_no_target.csv'
TEST_IMG_DIR = 'evaluation_dataset/evaluation_composite'
# Using the same Pseudo-Labels as Method 13 (Teacher M11) to ensure fair comparison
PSEUDO_LABEL_FILE = 'submission_final.csv' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Best Hyperparameters (Robust) ---
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.016,
    'num_leaves': 66,
    'feature_fraction': 0.93,
    'bagging_fraction': 0.90,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'colsample_bytree': 0.8
}

def get_resnet_embedder():
    print("Loading ResNet18...")
    resnet = models.resnet18(pretrained=True)
    modules = list(resnet.children())[:-1]
    embedder = torch.nn.Sequential(*modules).to(DEVICE)
    embedder.eval()
    return embedder

def extract_features(df, image_dir, embedder):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    embeddings = []
    
    for idx, row in df.iterrows():
        s2_filename = row.get('sentinel2_tiff_file_name')
        s2_path = os.path.join(image_dir, s2_filename) if isinstance(s2_filename, str) else None
        
        emb_vector = None
        if s2_path and os.path.exists(s2_path):
            try:
                with rasterio.open(s2_path) as src:
                    r, g, b = src.read(4), src.read(3), src.read(2)
                    def norm(x):
                        x = x.astype(float)
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

def add_spatial_encoding(train_df, test_df, pseudo_df=None):
    print("Adding Spatial Target Encodings...")
    
    # We only want to learn the encoding from REAL Train data to prevent bias/leakage from pseudo
    # Target Encoding columns
    target_cols = ['geolocation_name', 'country', 'region_economic_classification']
    
    # Prepare Train (True Target)
    train_df['log_target'] = np.log1p(train_df['construction_cost_per_m2_usd'])
    
    # 1. Generate OOF Encodings for Train Data (Prevent Self-Leakage)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for col in target_cols:
        train_df[f'{col}_target_enc'] = 0.0
        
        # OOF Loop
        for tr_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            
            # Global Mean of this fold
            global_mean = X_tr['log_target'].mean()
            
            # Map mean
            mapping = X_tr.groupby(col)['log_target'].mean()
            
            # Apply to Val (Fill Missing with Global Mean)
            # This simulates "unseen" data for the model during training
            train_df.loc[val_idx, f'{col}_target_enc'] = X_val[col].map(mapping).fillna(global_mean)
            
    # 2. Generate Global Encodings for Test/Pseudo (Use All Real Train Data)
    # This is the "Production" mapping
    mappings = {}
    global_mean = train_df['log_target'].mean()
    
    for col in target_cols:
        mappings[col] = train_df.groupby(col)['log_target'].mean()
        
        # Apply to Test
        test_df[f'{col}_target_enc'] = test_df[col].map(mappings[col]).fillna(global_mean)
        
        # Apply to Pseudo (if exists)
        if pseudo_df is not None:
             pseudo_df[f'{col}_target_enc'] = pseudo_df[col].map(mappings[col]).fillna(global_mean)

    # Cleanup temporary target
    train_df = train_df.drop(columns=['log_target'])
    
    return train_df, test_df, pseudo_df

def train_spatial():
    # 1. Load Data
    print("Loading Data...")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    pseudo_df = pd.read_csv(PSEUDO_LABEL_FILE)
    
    # Prepare Pseudo Data (Attach labels)
    # Be careful not to merge before Feature Engineering
    if 'data_id' in test_df.columns and 'data_id' in pseudo_df.columns:
        # We need the full test_df features for pseudo_df
        pseudo_df_full = test_df.copy()
        # Map labels
        label_map = dict(zip(pseudo_df['data_id'], pseudo_df['construction_cost_per_m2_usd']))
        pseudo_df_full['construction_cost_per_m2_usd'] = pseudo_df_full['data_id'].map(label_map)
    else:
        # Fallback
        pseudo_df_full = test_df.copy()
        pseudo_df_full['construction_cost_per_m2_usd'] = pseudo_df['construction_cost_per_m2_usd']
    
    # 2. SPATIAL ENCODING (The new Magic)
    # Note: We pass pseudo_df_full separately to apply the map
    train_df, test_df, pseudo_df_full = add_spatial_encoding(train_df, test_df, pseudo_df_full)
    
    # 3. Extract Features (ResNet)
    embedder = get_resnet_embedder()
    
    print("--- Train Set Features ---")
    train_embs = extract_features(train_df, TRAIN_IMG_DIR, embedder)
    
    print("--- Test Set Features ---")
    test_embs = extract_features(test_df, TEST_IMG_DIR, embedder)
    
    # Note: Using Test features for Pseudo DataFrame as well (same images)
    pseudo_embs = test_embs
    
    # 4. PCA
    print("Applying PCA...")
    all_embs = np.vstack([train_embs, test_embs])
    pca = PCA(n_components=16, random_state=42)
    all_pca = pca.fit_transform(all_embs)
    
    train_pca = all_pca[:len(train_df)]
    test_pca = all_pca[len(train_df):]
    pseudo_pca = test_pca
    
    # 5. Dataframe Construction
    pca_cols = [f'emb_{i}' for i in range(16)]
    
    # Join PCA
    train_full = pd.concat([train_df, pd.DataFrame(train_pca, columns=pca_cols, index=train_df.index)], axis=1)
    test_full = pd.concat([test_df, pd.DataFrame(test_pca, columns=pca_cols, index=test_df.index)], axis=1)
    pseudo_full = pd.concat([pseudo_df_full, pd.DataFrame(pseudo_pca, columns=pca_cols, index=pseudo_df_full.index)], axis=1)
    
    # 6. Std Feature Engineering
    for df in [train_full, test_full, pseudo_full]:
        if 'quarter_label' in df.columns:
            df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if '-Q' in str(x) else 0)
        
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'sentinel2_tiff_file_name' and col != 'viirs_tiff_file_name':
                df[col] = df[col].astype('category')

    # 7. Training Prep
    target_col = 'construction_cost_per_m2_usd'
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', target_col]
    
    features = [c for c in train_full.columns if c not in drop_cols]
    print(f"Features ({len(features)}): {features}")
    
    # Combined Train + Pseudo
    X_train = train_full[features]
    y_train = np.log1p(train_full[target_col])
    
    X_pseudo = pseudo_full[features]
    y_pseudo = np.log1p(pseudo_full[target_col])
    
    X_combined = pd.concat([X_train, X_pseudo], axis=0)
    y_combined = pd.concat([y_train, y_pseudo], axis=0)
    
    print(f"Training on {len(X_combined)} samples (Real + Pseudo)...")
    
    train_data = lgb.Dataset(X_combined, label=y_combined)
    
    # 8. Train
    bst = lgb.train(params, train_data, num_boost_round=1000)
    
    # 9. Predict
    preds_log = bst.predict(test_full[features])
    preds = np.expm1(preds_log)
    
    sub = pd.DataFrame()
    if 'data_id' in test_df.columns:
        sub['data_id'] = test_df['data_id']
    sub['construction_cost_per_m2_usd'] = preds
    sub.to_csv('submission020.csv', index=False)
    print("Submission saved to submission020.csv")

if __name__ == "__main__":
    train_spatial()
