import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# --- Config ---
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 32 # Small batch size for mining
EPOCHS = 30
LR = 1e-4
EMBED_DIM = 512 # ResNet18 default

TRAIN_IMG_DIR = 'dataset/train_png'
TEST_IMG_DIR = 'evaluation_dataset/test_png'
TRAIN_CSV = 'dataset/train_tabular.csv'
TEST_CSV = 'evaluation_dataset/evaluation_tabular_no_target.csv'
SUBMISSION_FILE = 'submission032.csv'

# Set Seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# --- Dataset ---
class ConstructionDataset(Dataset):
    def __init__(self, csv_file, img_dir, is_train=True, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.is_train = is_train
        self.transform = transform
        
        # Prepare targets
        if self.is_train:
            self.labels = np.log1p(self.df['construction_cost_per_m2_usd'].values)
        else:
            self.labels = np.zeros(len(self.df)) # Dummy
            
        # File paths
        # Assuming we need to map IDs to filenames if not present.
        # But Phase 1 scripts showed filenames are likely in the CSV or derived.
        # Let's check columns first. Using 'data_id' to find file.
        self.ids = self.df['data_id'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # We need to find the file.
        # M30 util converted 'sentinel2_tiff_file_name' to PNG in a flat dir or similar.
        # Let's assume standard 'data_id.png' or we lookup filename.
        # Let's use the 'sentinel2_tiff_file_name' col but replace extension.
        
        row = self.df.iloc[idx]
        if 'sentinel2_tiff_file_name' in row:
             fname = row['sentinel2_tiff_file_name'].replace('.tif', '') + '.png'
        else:
             # Fallback
             fname = f"{row['data_id']}.png"

        img_path = os.path.join(self.img_dir, fname)
        
        # Robust loading
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Create black image if missing (robustness)
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
            
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return image, label, idx

# --- Model ---
class PriceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # Remove fc
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return F.normalize(features, p=2, dim=1) # L2 Normalize

# --- Loss (Online Triplet) ---
def triplet_loss(features, labels, margin=0.5):
    """
    Online mining:
    For each item in batch, find:
    - Positive: Item with closest price (but not self).
    - Negative: Item with furthest price.
    Actually for REGRESSION metric learning:
    - We want distance(i, j) to be proportional to abs(price_i - price_j).
    - Or simpler: Positive if abs(diff) < small_eps, Negative if abs(diff) > big_eps.
    Let's go with "Contrastive Regression Loss":
    Maximize similarity if diff is small, Minimize if diff is large.
    """
    
    bs = features.size(0)
    loss = 0.0
    valid_triplets = 0
    
    # Distance matrix of embeddings
    # d(u,v)^2 = 2 - 2u*v (since normalized)
    sim_matrix = torch.matmul(features, features.T) # (B, B) range [-1, 1]
    
    # Distance matrix of targets
    # (B, 1) - (1, B) -> (B, B)
    price_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
    
    # Hard Thresholds for mining
    POS_THRESH = 0.1 # Very similar price
    NEG_THRESH = 0.5 # Validly different price
    
    for i in range(bs):
        # Positives: diff < POS_THRESH
        # Negatives: diff > NEG_THRESH
        
        pos_mask = (price_diff[i] < POS_THRESH) & (torch.arange(bs).to(DEVICE) != i)
        neg_mask = (price_diff[i] > NEG_THRESH)
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            # Hardest Positive: Lowest Similarity among positives
            # Hardest Negative: Highest Similarity among negatives
            
            hard_pos_sim = sim_matrix[i, pos_mask].min()
            hard_neg_sim = sim_matrix[i, neg_mask].max()
            
            # Loss: We want pos_sim > neg_sim + margin
            l = F.relu(hard_neg_sim - hard_pos_sim + margin)
            loss += l
            valid_triplets += 1
            
    if valid_triplets > 0:
        return loss / valid_triplets
    return torch.tensor(0.0, device=DEVICE, requires_grad=True)

# --- Functions ---
def train_model():
    # Transforms
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Data
    train_ds = ConstructionDataset(TRAIN_CSV, TRAIN_IMG_DIR, is_train=True, transform=train_tfm)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = PriceEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("Starting Metric Learning (SupCon)...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        steps = 0
        for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            feats = model(imgs) # (B, 512)
            loss = triplet_loss(feats, labels)
            
            if loss.item() > 1e-6: # Skip 0 loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1
        
        if steps > 0:
            print(f"Epoch {epoch+1}: Loss = {epoch_loss/steps:.4f}")
        else:
             print(f"Epoch {epoch+1}: Loss = 0.0000 (No hard triplets found)")
             
    return model

def extract_features(model, img_dir, csv_file, is_train=False):
    model.eval()
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = ConstructionDataset(csv_file, img_dir, is_train=is_train, transform=tfm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4)
    
    all_feats = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            feats = model(imgs)
            all_feats.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.vstack(all_feats), np.array(all_labels)

def predict_knn(train_feats, train_labels, test_feats, test_csv, k=50):
    print(f"Running KNN (K={k})...")
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(train_feats)
    
    dists, indices = knn.kneighbors(test_feats)
    
    preds = []
    for i in range(len(test_feats)):
        # Weighted Average
        # M21 logic: weights = 1/distance
        # Cosine distance is in [0, 2]. 0 is identical.
        d = dists[i]
        d = np.maximum(d, 1e-6) # Avoid div0
        weights = 1.0 / d
        
        # Neighbor values
        neighbor_vals = train_labels[indices[i]]
        
        # Weighted Mean
        pred = np.sum(weights * neighbor_vals) / np.sum(weights)
        preds.append(pred)
        
    return np.expm1(np.array(preds))

if __name__ == "__main__":
    # 1. Train
    model = train_model()
    
    # 2. Extract
    print("Extracting Train Embeddings...")
    X_train, y_train_log = extract_features(model, TRAIN_IMG_DIR, TRAIN_CSV, is_train=True)
    
    print("Extracting Test Embeddings...")
    X_test, _ = extract_features(model, TEST_IMG_DIR, TEST_CSV, is_train=False)
    
    # 3. Predict
    preds = predict_knn(X_train, y_train_log, X_test, TEST_CSV)
    
    # 4. Save
    df_test = pd.read_csv(TEST_CSV)
    sub = pd.DataFrame({'data_id': df_test['data_id'], 'construction_cost_per_m2_usd': preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved {SUBMISSION_FILE}")
