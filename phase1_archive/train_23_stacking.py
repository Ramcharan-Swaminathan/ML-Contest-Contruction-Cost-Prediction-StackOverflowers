import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import os
import rasterio
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
TRAIN_DATA = 'dataset/train_tabular.csv'
TRAIN_IMG_DIR = 'dataset/train_composite'
TEST_DATA = 'evaluation_tabular_no_target.csv'
TEST_IMG_DIR = 'evaluation_dataset/evaluation_composite'
PSEUDO_LABEL_FILE = 'submission_final.csv' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Hyperparameters ---
# M21 (Champion) Params
lgb_params_m21 = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'learning_rate': 0.016, 'num_leaves': 66, 'feature_fraction': 0.93, 
    'bagging_fraction': 0.90, 'bagging_freq': 5, 'min_child_samples': 20,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'colsample_bytree': 0.8
}

# M09 (Robust Baseline) Params
lgb_params_m09 = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'learning_rate': 0.02, 'num_leaves': 31, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 1
}

# CatBoost Params
cat_params = {
    'iterations': 1000, 'learning_rate': 0.03, 'depth': 6, 'loss_function': 'RMSE',
    'verbose': 0, 'allow_writing_files': False
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
                    # TTA 2x (Speed up Stacking)
                    rot_embs = []
                    for angle in [0, 180]:
                        t_img = img.rotate(angle) if angle > 0 else img
                        t_tensor = preprocess(t_img).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            rot_embs.append(embedder(t_tensor).cpu().numpy().flatten())
                    emb_vector = np.mean(rot_embs, axis=0)
            except: pass
        if emb_vector is None: emb_vector = np.zeros(512)
        embeddings.append(emb_vector)
    return np.array(embeddings)

def add_spatial_encoding(train_df, test_df):
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
    train_df = train_df.drop(columns=['log_target'])
    return train_df, test_df

def add_visual_knn(train_df, test_df, train_embs, test_embs, k=50):
    print("Adding Visual KNN...")
    y_train = np.log1p(train_df['construction_cost_per_m2_usd'].values)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn.fit(train_embs)
    dists, indices = nn.kneighbors(train_embs, n_neighbors=k+1)
    train_knn = []
    for i in range(len(train_embs)):
        neighbor_idxs = indices[i, 1:]
        train_knn.append(np.mean(y_train[neighbor_idxs]))
    train_df['visual_knn_cost'] = train_knn
    
    dists, indices = nn.kneighbors(test_embs, n_neighbors=k)
    test_knn = []
    for i in range(len(test_embs)):
        test_knn.append(np.mean(y_train[indices[i]]))
    test_df['visual_knn_cost'] = test_knn
    return train_df, test_df

def train_stacking():
    print("Loading Data...")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    # 1. Feature Engineering (Shared)
    train_df, test_df = add_spatial_encoding(train_df, test_df)
    
    embedder = get_resnet_embedder()
    print("Extracting Embeddings...")
    train_embs = extract_features(train_df, TRAIN_IMG_DIR, embedder)
    test_embs = extract_features(test_df, TEST_IMG_DIR, embedder)
    
    train_df, test_df = add_visual_knn(train_df, test_df, train_embs, test_embs)
    
    print("PCA...")
    pca = PCA(n_components=16, random_state=42)
    all_pca = pca.fit_transform(np.vstack([train_embs, test_embs]))
    train_pca = all_pca[:len(train_df)]
    test_pca = all_pca[len(train_df):]
    pca_cols = [f'emb_{i}' for i in range(16)]
    train_df = pd.concat([train_df, pd.DataFrame(train_pca, columns=pca_cols, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(test_pca, columns=pca_cols, index=test_df.index)], axis=1)
    
    # Prep for Models
    for df in [train_df, test_df]:
        if 'quarter_label' in df.columns:
            df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if '-Q' in str(x) else 0)
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'sentinel2_tiff_file_name' and col != 'viirs_tiff_file_name':
                df[col] = df[col].astype('category')

    target_col = 'construction_cost_per_m2_usd'
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', target_col]
    features = [c for c in train_df.columns if c not in drop_cols]
    cat_features = [c for c in features if train_df[c].dtype.name == 'category']
    print(f"Features: {features}")
    
    X = train_df[features]
    y = np.log1p(train_df[target_col])
    X_test = test_df[features]
    
    # Fill NA for CatBoost
    X_cat = X.copy()
    X_test_cat = X_test.copy()
    for c in cat_features:
        X_cat[c] = X_cat[c].cat.add_categories('Missing').fillna('Missing')
        X_test_cat[c] = X_test_cat[c].cat.add_categories('Missing').fillna('Missing')
    
    # --- STACKING LOOP ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # OOF Arrays
    oof_m21 = np.zeros(len(X))
    oof_m09 = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    
    test_m21 = np.zeros(len(X_test))
    test_m09 = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))
    
    print("Starting Stacking CV...")
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold+1}/5")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        X_tr_cat, X_val_cat = X_cat.iloc[tr_idx], X_cat.iloc[val_idx]
        
        # M21 (Champion LGBM)
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)
        bst_m21 = lgb.train(lgb_params_m21, dtrain, num_boost_round=1000, valid_sets=[dval], 
                            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        oof_m21[val_idx] = bst_m21.predict(X_val)
        test_m21 += bst_m21.predict(X_test) / 5
        
        # M09 (Baseline LGBM - Different params)
        bst_m09 = lgb.train(lgb_params_m09, dtrain, num_boost_round=1000, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        oof_m09[val_idx] = bst_m09.predict(X_val)
        test_m09 += bst_m09.predict(X_test) / 5
        
        # CatBoost
        train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val_cat, y_val, cat_features=cat_features)
        cbst = CatBoostRegressor(**cat_params)
        cbst.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
        oof_cat[val_idx] = cbst.predict(X_val_cat)
        test_cat += cbst.predict(X_test_cat) / 5
        
    print(f"OOF Scores - M21: {np.mean((oof_m21 - y)**2)**0.5:.5f}, M09: {np.mean((oof_m09 - y)**2)**0.5:.5f}, Cat: {np.mean((oof_cat - y)**2)**0.5:.5f}")
    
    # --- META LEARNER ---
    print("Training Meta Learner (Ridge)...")
    X_oof = np.vstack([oof_m21, oof_m09, oof_cat]).T
    X_meta_test = np.vstack([test_m21, test_m09, test_cat]).T
    
    meta = Ridge(alpha=1.0)
    meta.fit(X_oof, y)
    
    print(f"Meta Weights: M21={meta.coef_[0]:.4f}, M09={meta.coef_[1]:.4f}, Cat={meta.coef_[2]:.4f}")
    
    final_preds_log = meta.predict(X_meta_test)
    final_preds = np.expm1(final_preds_log)
    
    sub = pd.DataFrame()
    if 'data_id' in test_df.columns:
        sub['data_id'] = test_df['data_id']
    sub['construction_cost_per_m2_usd'] = final_preds
    sub.to_csv('submission023.csv', index=False)
    print("Submission saved to submission023.csv")

if __name__ == "__main__":
    train_stacking()
