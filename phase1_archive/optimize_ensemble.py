import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import rasterio
from PIL import Image
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA

# --- Config ---
DATA_PATH = 'dataset/train_tabular.csv'
IMAGE_DIR = 'dataset/train_composite'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Reconstruct Validation Set ---
print("Reconstructing Validation Set (Random State 42)...")
df = pd.read_csv(DATA_PATH)
_, val_df = train_test_split(df, test_size=0.2, random_state=42)
y_val = np.log1p(val_df['construction_cost_per_m2_usd'].values)
print(f"Validation Set Size: {len(val_df)}")

# --- 2. Helper: Image Loading ---
def load_image(row, image_dir):
    s2_filename = row.get('sentinel2_tiff_file_name')
    s2_path = os.path.join(image_dir, s2_filename) if isinstance(s2_filename, str) else None
    
    if s2_path and os.path.exists(s2_path):
        try:
            with rasterio.open(s2_path) as src:
                r, g, b = src.read(4), src.read(3), src.read(2)
                def norm(band):
                    band = band.astype(float)
                    p98 = np.percentile(band, 98)
                    if p98 > 0: band = np.clip(band, 0, p98) / p98
                    else: band = np.clip(band, 0, 1)
                    return (band * 255).astype(np.uint8)
                rgb = np.dstack((norm(r), norm(g), norm(b)))
                return Image.fromarray(rgb)
        except: pass
    return Image.new('RGB', (224, 224), (0,0,0))

# --- 3. Method 10 Predictions (DL) ---
print("Generating Method 10 Predictions (DL)...")
class ConstructionNet(nn.Module):
    def __init__(self):
        super(ConstructionNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
    def forward(self, x): return self.resnet(x)

dl_model = ConstructionNet().to(DEVICE)
dl_model.load_state_dict(torch.load('model_checkpoint_10.pth', map_location=DEVICE))
dl_model.eval()

dl_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dl_preds = []
with torch.no_grad():
    for _, row in val_df.iterrows():
        img = load_image(row, IMAGE_DIR)
        # TTA 4x
        preds = []
        for angle in [0, 90, 180, 270]:
            t_img = dl_transform(img.rotate(angle) if angle > 0 else img).unsqueeze(0).to(DEVICE)
            preds.append(dl_model(t_img).item())
        dl_preds.append(np.mean(preds))
dl_preds = np.array(dl_preds)

print(f"Method 10 Val RMSLE: {np.sqrt(mean_squared_error(y_val, dl_preds)):.5f}")

# --- 4. Method 09 Predictions (LGBM) ---
print("Generating Method 09 Predictions (LGBM)...")

# Need to re-extract PCA features
# Load PCA
with open('pca_model_09.pkl', 'rb') as f:
    pca = pickle.load(f)

# Feature Extractor for M09
resnet = models.resnet18(pretrained=True)
embedder = nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
embedder.eval()

emb_transform = transforms.Compose([
    transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

embeddings = []
with torch.no_grad():
    for _, row in val_df.iterrows():
        img = load_image(row, IMAGE_DIR)
        rot_embs = []
        for angle in [0, 90, 180, 270]:
            t_img = emb_transform(img.rotate(angle) if angle > 0 else img).unsqueeze(0).to(DEVICE)
            rot_embs.append(embedder(t_img).cpu().numpy().flatten())
        embeddings.append(np.mean(rot_embs, axis=0))

embeddings = np.array(embeddings)
pca_feats = pca.transform(embeddings)

# Create DataFrame for LGBM
emb_cols = [f'emb_{i}' for i in range(16)]
emb_df = pd.DataFrame(pca_feats, columns=emb_cols, index=val_df.index)
val_processed = pd.concat([val_df, emb_df], axis=1)

# Feature Steps (match train_09)
if 'quarter_label' in val_processed.columns:
    val_processed['quarter'] = val_processed['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if '-Q' in str(x) else 0)

lgbm_model = lgb.Booster(model_file='model_checkpoint_09.txt')
features = lgbm_model.feature_name()

# Ensure category consistency
for col in val_processed.columns:
    if val_processed[col].dtype == 'object':
        val_processed[col] = val_processed[col].astype('category')

lgb_preds = lgbm_model.predict(val_processed[features])
# Note: M09 predicts log1p directly if trained on log1p

print(f"Method 09 Val RMSLE: {np.sqrt(mean_squared_error(y_val, lgb_preds)):.5f}")

# --- 5. Optimize Weights ---
# Objective: Minimize RMSLE of w * P9 + (1-w) * P10
def objective(w):
    blend = w * lgb_preds + (1 - w) * dl_preds
    return np.sqrt(mean_squared_error(y_val, blend))

res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
best_w = res.x
best_score = res.fun

print(f"\n--- Optimization Results ---")
print(f"Best Weight for Method 09 (LGBM): {best_w:.4f}")
print(f"Best Weight for Method 10 (DL):   {1 - best_w:.4f}")
print(f"Best Ensemble RMSLE: {best_score:.5f}")

# --- 6. Save Weights ---
with open('ensemble_weights.txt', 'w') as f:
    f.write(f"{best_w}")
