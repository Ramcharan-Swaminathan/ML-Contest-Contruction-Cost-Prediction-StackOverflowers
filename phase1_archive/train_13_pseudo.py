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
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
TRAIN_DATA = 'dataset/train_tabular.csv'
TRAIN_IMG_DIR = 'dataset/train_composite'
TEST_DATA = 'evaluation_tabular_no_target.csv'
TEST_IMG_DIR = 'evaluation_dataset/evaluation_composite'
PSEUDO_LABEL_FILE = 'submission_final.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Best Hyperparameters from Method 09 ---
# As documented in Walkthrough
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
    'colsample_bytree': 0.8 # Added for robustness
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
    print(f"Extracting features for {len(df)} images...")
    
    for idx, row in df.iterrows():
        if idx % 100 == 0: print(f"  {idx}/{len(df)}")
        
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

def train_pseudo():
    # 1. Load Data
    print("Loading Data...")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    pseudo_df = pd.read_csv(PSEUDO_LABEL_FILE)
    
    # Check alignment
    # We assume pseudo_df and test_df are aligned or join on data_id
    if 'data_id' in test_df.columns and 'data_id' in pseudo_df.columns:
        test_df = test_df.merge(pseudo_df[['data_id', 'construction_cost_per_m2_usd']], on='data_id', how='left')
    else:
        # Fallback: simple assignment if lengths match (risky but usually fine in competitions)
        test_df['construction_cost_per_m2_usd'] = pseudo_df['construction_cost_per_m2_usd']
    
    # 2. Extract Features
    embedder = get_resnet_embedder()
    
    print("--- Train Set ---")
    train_embs = extract_features(train_df, TRAIN_IMG_DIR, embedder)
    
    print("--- Test Set (Pseudo) ---")
    test_embs = extract_features(test_df, TEST_IMG_DIR, embedder)
    
    # 3. PCA on Combined Data
    print("Applying PCA...")
    all_embs = np.vstack([train_embs, test_embs])
    pca = PCA(n_components=16, random_state=42)
    all_pca = pca.fit_transform(all_embs)
    
    train_pca = all_pca[:len(train_df)]
    test_pca = all_pca[len(train_df):]
    
    # 4. Prepare Tabular + PCA DataFrame
    pca_cols = [f'emb_{i}' for i in range(16)]
    
    train_pca_df = pd.DataFrame(train_pca, columns=pca_cols, index=train_df.index)
    test_pca_df = pd.DataFrame(test_pca, columns=pca_cols, index=test_df.index)
    
    train_full = pd.concat([train_df, train_pca_df], axis=1)
    test_full = pd.concat([test_df, test_pca_df], axis=1)
    
    # 5. Feature Engineering (Match previous methods)
    for df in [train_full, test_full]:
        if 'quarter_label' in df.columns:
            df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if '-Q' in str(x) else 0)
        
        # Categoricals
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'sentinel2_tiff_file_name' and col != 'viirs_tiff_file_name':
                df[col] = df[col].astype('category')

    # 6. Prepare Matrix
    target_col = 'construction_cost_per_m2_usd'
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', target_col]
    
    features = [c for c in train_full.columns if c not in drop_cols]
    
    # Combine Train + Test for Training
    X_train = train_full[features]
    y_train = np.log1p(train_full[target_col])
    
    X_test_pseudo = test_full[features]
    y_test_pseudo = np.log1p(test_full[target_col])
    
    X_combined = pd.concat([X_train, X_test_pseudo], axis=0)
    y_combined = pd.concat([y_train, y_test_pseudo], axis=0)
    
    print(f"Training on combined dataset: {len(X_combined)} rows (Train: {len(X_train)} + Pseudo: {len(X_test_pseudo)})")
    
    train_data = lgb.Dataset(X_combined, label=y_combined)
    
    # 7. Train Model
    bst = lgb.train(params, train_data, num_boost_round=1000)
    
    # 8. Predict (yes, predicting on the same data we trained on - for pseudo labeling strat, this is standard final step)
    # Actually, we just need to predict on X_test_pseudo for submission
    preds_log = bst.predict(X_test_pseudo)
    preds = np.expm1(preds_log)
    
    # 9. Save
    sub = pd.DataFrame()
    if 'data_id' in test_df.columns:
        sub['data_id'] = test_df['data_id']
    sub['construction_cost_per_m2_usd'] = preds
    sub.to_csv('submission013.csv', index=False)
    print("Submission saved to submission013.csv")

if __name__ == "__main__":
    train_pseudo()
