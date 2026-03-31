import pandas as pd
import numpy as np
import os
import rasterio
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor, Pool
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
TRAIN_DATA = 'dataset/train_tabular.csv'
TRAIN_IMG_DIR = 'dataset/train_composite'
TEST_DATA = 'evaluation_tabular_no_target.csv'
TEST_IMG_DIR = 'evaluation_dataset/evaluation_composite'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Feature Extraction (Reusing Method 09 Pipeline) ---
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
    # print(f"Extracting features for {len(df)} images...")
    
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

def train_catboost():
    # 1. Load Data
    print("Loading Data...")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    # 2. Extract Features (or check if cached)
    # Ideally should cache, but for script simplicity we re-run (takes ~2 mins)
    embedder = get_resnet_embedder()
    
    print("Extracting Train Features...")
    train_embs = extract_features(train_df, TRAIN_IMG_DIR, embedder)
    
    print("Extracting Test Features...")
    test_embs = extract_features(test_df, TEST_IMG_DIR, embedder)
    
    # 3. PCA
    print("Applying PCA...")
    pca = PCA(n_components=16, random_state=42)
    train_pca = pca.fit_transform(train_embs)
    test_pca = pca.transform(test_embs)
    
    pca_cols = [f'emb_{i}' for i in range(16)]
    train_pca_df = pd.DataFrame(train_pca, columns=pca_cols, index=train_df.index)
    test_pca_df = pd.DataFrame(test_pca, columns=pca_cols, index=test_df.index)
    
    train_full = pd.concat([train_df, train_pca_df], axis=1)
    test_full = pd.concat([test_df, test_pca_df], axis=1)
    
    # 4. Feature Engineering
    for df in [train_full, test_full]:
        if 'quarter_label' in df.columns:
            df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if '-Q' in str(x) else 0)
    
    # 5. Setup CatBoost
    target_col = 'construction_cost_per_m2_usd'
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', target_col]
    
    features = [c for c in train_full.columns if c not in drop_cols]
    cat_features = [c for c in features if train_full[c].dtype == 'object' or train_full[c].dtype.name == 'category']
    
    print(f"Features: {len(features)}")
    print(f"Categorical Features: {cat_features}")
    
    # CatBoost Strict Mode Fix: Fill NaNs in categoricals
    for col in cat_features:
        train_full[col] = train_full[col].astype(str).fillna("Missing")
        test_full[col] = test_full[col].astype(str).fillna("Missing")
    
    X = train_full[features]
    y = np.log1p(train_full[target_col])
    
    X_test = test_full[features]
    
    # 6. Train with Cross-Validation (to get reliable score)
    # CatBoost handles categorical features natively, very powerful
    
    params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'iterations': 2000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 50,
        'verbose': 200,
        'cat_features': cat_features
    }
    
    print("Training CatBoost...")
    model = CatBoostRegressor(**params)
    model.fit(X, y, verbose=200)
    
    # 7. Validation Score (Self-reported by model on training isn't enough, need CV?)
    # For now, let's just create the prediction file
    
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    
    sub = pd.DataFrame()
    if 'data_id' in test_df.columns:
        sub['data_id'] = test_df['data_id']
    sub['construction_cost_per_m2_usd'] = preds
    sub.to_csv('submission015.csv', index=False)
    print("Submission saved to submission015.csv")
    
    # Save Model
    model.save_model("catboost_model.cbm")

if __name__ == "__main__":
    train_catboost()
