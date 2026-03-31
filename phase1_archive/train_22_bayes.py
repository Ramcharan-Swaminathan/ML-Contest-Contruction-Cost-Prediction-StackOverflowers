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
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
TRAIN_DATA = 'dataset/train_tabular.csv'
TRAIN_IMG_DIR = 'dataset/train_composite'
TEST_DATA = 'evaluation_tabular_no_target.csv'
TEST_IMG_DIR = 'evaluation_dataset/evaluation_composite'
PSEUDO_LABEL_FILE = 'submission_final.csv' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Params ---
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
        if emb_vector is None: emb_vector = np.zeros(512)
        embeddings.append(emb_vector)
    return np.array(embeddings)

def bayesian_target_encoding(train_df, test_df, pseudo_df, col, target_col, weight=10):
    """
    Computes a smoothed mean: (n * mean + w * global_mean) / (n + w)
    """
    global_mean = train_df[target_col].mean()
    
    # --- 1. OOF for Train ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df[f'{col}_bayes_enc'] = 0.0
    
    for tr_idx, val_idx in kf.split(train_df):
        X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        
        # Stats on training split
        agg = X_tr.groupby(col)[target_col].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        
        # Bayesian Smooth
        smooth = (counts * means + weight * global_mean) / (counts + weight)
        
        # Map
        train_df.loc[val_idx, f'{col}_bayes_enc'] = X_val[col].map(smooth).fillna(global_mean)
        
    # --- 2. Global Map for Test/Pseudo ---
    # Using FULL train data
    agg = train_df.groupby(col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + weight * global_mean) / (counts + weight)
    
    test_df[f'{col}_bayes_enc'] = test_df[col].map(smooth).fillna(global_mean)
    if pseudo_df is not None:
        pseudo_df[f'{col}_bayes_enc'] = pseudo_df[col].map(smooth).fillna(global_mean)
        
    return train_df, test_df, pseudo_df

def add_visual_knn_feature(train_df, test_df, pseudo_df, train_embs, test_embs, k=50):
    # Same as Method 21
    print(f"Adding Visual KNN Feature (K={k})...")
    y_train = np.log1p(train_df['construction_cost_per_m2_usd'].values)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn.fit(train_embs)
    
    # Train OOF
    dists, indices = nn.kneighbors(train_embs, n_neighbors=k+1)
    train_knn = []
    for i in range(len(train_embs)):
        neighbor_idxs = indices[i, 1:]
        mean_val = np.mean(y_train[neighbor_idxs])
        train_knn.append(mean_val)
    train_df['visual_knn_cost'] = train_knn
    
    # Test
    dists, indices = nn.kneighbors(test_embs, n_neighbors=k)
    test_knn = []
    for i in range(len(test_embs)):
        neighbor_idxs = indices[i]
        mean_val = np.mean(y_train[neighbor_idxs])
        test_knn.append(mean_val)
    test_df['visual_knn_cost'] = test_knn
    pseudo_df['visual_knn_cost'] = test_knn
    
    return train_df, test_df, pseudo_df

def train_bayes():
    print("Loading Data...")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    pseudo_df = pd.read_csv(PSEUDO_LABEL_FILE)
    
    if 'data_id' in test_df.columns and 'data_id' in pseudo_df.columns:
        pseudo_df_full = test_df.copy()
        label_map = dict(zip(pseudo_df['data_id'], pseudo_df['construction_cost_per_m2_usd']))
        pseudo_df_full['construction_cost_per_m2_usd'] = pseudo_df_full['data_id'].map(label_map)
    else:
        pseudo_df_full = test_df.copy()
        pseudo_df_full['construction_cost_per_m2_usd'] = pseudo_df['construction_cost_per_m2_usd']
    
    # Prep Log Target for Encoding
    train_df['log_target'] = np.log1p(train_df['construction_cost_per_m2_usd'])
    
    # 1. BAYESIAN ENCODING (Replaces Simple Mean Encoding)
    print("Applying Bayesian Target Encoding...")
    target_cols = ['geolocation_name', 'country', 'region_economic_classification']
    for col in target_cols:
        # Weight=10 means we need ~10 samples to trust the local mean 50/50
        train_df, test_df, pseudo_df_full = bayesian_target_encoding(train_df, test_df, pseudo_df_full, col, 'log_target', weight=10)
    
    train_df = train_df.drop(columns=['log_target'])
    
    # 2. Extract Embeddings
    embedder = get_resnet_embedder()
    print("Extracting Embeddings...")
    train_embs = extract_features(train_df, TRAIN_IMG_DIR, embedder)
    test_embs = extract_features(test_df, TEST_IMG_DIR, embedder)
    
    # 3. Visual KNN
    train_df, test_df, pseudo_df_full = add_visual_knn_feature(
        train_df, test_df, pseudo_df_full, 
        train_embs, test_embs, k=50
    )
    
    # 4. PCA
    print("Applying PCA...")
    all_embs = np.vstack([train_embs, test_embs])
    pca = PCA(n_components=16, random_state=42)
    all_pca = pca.fit_transform(all_embs)
    train_pca = all_pca[:len(train_df)]
    test_pca = all_pca[len(train_df):]
    
    pca_cols = [f'emb_{i}' for i in range(16)]
    train_full = pd.concat([train_df, pd.DataFrame(train_pca, columns=pca_cols, index=train_df.index)], axis=1)
    test_full = pd.concat([test_df, pd.DataFrame(test_pca, columns=pca_cols, index=test_df.index)], axis=1)
    pseudo_full = pd.concat([pseudo_df_full, pd.DataFrame(test_pca, columns=pca_cols, index=pseudo_df_full.index)], axis=1)
    
    # 5. Type Conversions
    for df in [train_full, test_full, pseudo_full]:
        if 'quarter_label' in df.columns:
            df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if '-Q' in str(x) else 0)
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'sentinel2_tiff_file_name' and col != 'viirs_tiff_file_name':
                df[col] = df[col].astype('category')

    target_col = 'construction_cost_per_m2_usd'
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', target_col]
    features = [c for c in train_full.columns if c not in drop_cols]
    print(f"Features: {features}")
    
    X_train = train_full[features]
    y_train = np.log1p(train_full[target_col])
    X_pseudo = pseudo_full[features]
    y_pseudo = np.log1p(pseudo_full[target_col])
    
    X = pd.concat([X_train, X_pseudo], axis=0)
    y = pd.concat([y_train, y_pseudo], axis=0)
    
    print(f"Training on {len(X)} samples...")
    train_data = lgb.Dataset(X, label=y)
    bst = lgb.train(params, train_data, num_boost_round=1000)
    
    preds_log = bst.predict(test_full[features])
    preds = np.expm1(preds_log)
    
    sub = pd.DataFrame()
    if 'data_id' in test_df.columns:
        sub['data_id'] = test_df['data_id']
    sub['construction_cost_per_m2_usd'] = preds
    sub.to_csv('submission022.csv', index=False)
    print("Done. Saved submission022.csv")

if __name__ == "__main__":
    train_bayes()
