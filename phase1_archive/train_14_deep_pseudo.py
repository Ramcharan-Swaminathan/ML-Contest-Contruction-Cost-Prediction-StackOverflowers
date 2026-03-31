import pandas as pd
import numpy as np
import os
import argparse
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import copy
import time
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
# User has a GPU (CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 150  # Enough for convergence with more data
IMAGE_SIZE = 224

TRAIN_DATA = 'dataset/train_tabular.csv'
TRAIN_IMG_DIR = 'dataset/train_composite'
TEST_DATA = 'evaluation_tabular_no_target.csv'
TEST_IMG_DIR = 'evaluation_dataset/evaluation_composite'
PSEUDO_LABEL_FILE = 'submission013.csv'  # Features from Method 13

# --- Dataset ---
class ConstructionDataset(Dataset):
    def __init__(self, df, image_source_map, transform=None):
        """
        image_source_map: dict mapping 'filename' -> 'directory_path'
        Pre-loads all images into RAM for speed.
        """
        self.df = df
        self.image_source_map = image_source_map
        self.transform = transform
        self.images = {}
        
        # Limit RAM usage: Only cache the first 25% of images
        cache_limit = int(len(df) * 0.25)
        print(f"Pre-loading {cache_limit} images (25%) into RAM for hybrid speed...")
        
        for idx in range(len(df)):
            if idx >= cache_limit:
                break
                
            row = df.iloc[idx]
            if idx % 500 == 0: print(f"  Loaded {idx}/{cache_limit}")
            s2_filename = row.get('sentinel2_tiff_file_name')
            image_dir = image_source_map.get(s2_filename, TRAIN_IMG_DIR)
            s2_path = os.path.join(image_dir, s2_filename) if isinstance(s2_filename, str) else None
            
            image = None
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
                        image = Image.fromarray(rgb)
                except: pass
            
            if image is None:
                image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(0,0,0))
                
            self.images[idx] = image
        print("Partial Image Loading Complete.")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = np.log1p(row['construction_cost_per_m2_usd'])
        
        # Check Cache
        if idx in self.images:
            image = self.images[idx]
        else:
            # Load from Disk on Demand
            s2_filename = row.get('sentinel2_tiff_file_name')
            image_dir = self.image_source_map.get(s2_filename, TRAIN_IMG_DIR)
            s2_path = os.path.join(image_dir, s2_filename) if isinstance(s2_filename, str) else None
            
            image = None
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
                        image = Image.fromarray(rgb)
                except: pass
            
            if image is None:
                image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(0,0,0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(target, dtype=torch.float32)

# --- Model ---
class ConstructionNet(nn.Module):
    def __init__(self):
        super(ConstructionNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # Unfreeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = True
            
        # Regression Head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.resnet(x)

def train_and_predict():
    # 1. Prepare Data
    print("Loading Real Data...")
    real_df = pd.read_csv(TRAIN_DATA)
    
    # Validation Split (CRITICAL: Must be Real Data Only)
    train_real_df, val_real_df = train_test_split(real_df, test_size=0.2, random_state=42)
    
    print("Loading Pseudo Data...")
    pseudo_labels = pd.read_csv(PSEUDO_LABEL_FILE)
    test_df = pd.read_csv(TEST_DATA)
    
    # Merge targets onto test data
    if 'data_id' in test_df.columns and 'data_id' in pseudo_labels.columns:
        pseudo_df = test_df.merge(pseudo_labels[['data_id', 'construction_cost_per_m2_usd']], on='data_id', how='left')
    else:
        pseudo_df = test_df.copy()
        pseudo_df['construction_cost_per_m2_usd'] = pseudo_labels['construction_cost_per_m2_usd']
    
    # Combine Train Real + Train Pseudo
    train_combined_df = pd.concat([train_real_df, pseudo_df], axis=0).reset_index(drop=True)
    
    print(f"Dataset Stats:")
    print(f"  Real Train:   {len(train_real_df)}")
    print(f"  Pseudo Train: {len(pseudo_df)}")
    print(f"  Combined:     {len(train_combined_df)}")
    print(f"  Validation:   {len(val_real_df)} (Real Data Only)")
    
    # Image Source Mapping (Filename -> Directory)
    # This assumes filenames are unique across sets, which they usually are in Kaggle
    img_source = {}
    for fname in real_df['sentinel2_tiff_file_name'].dropna():
        img_source[fname] = TRAIN_IMG_DIR
    for fname in pseudo_df['sentinel2_tiff_file_name'].dropna():
        img_source[fname] = TEST_IMG_DIR
        
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ConstructionDataset(train_combined_df, img_source, transform=train_transform)
    val_dataset = ConstructionDataset(val_real_df, img_source, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Setup Model
    print("Initializing ResNet18...")
    model = ConstructionNet().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"Starting Training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- Train Loop ---
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_dataset)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
        
        val_loss = val_loss / len(val_dataset)
        val_rmse = np.sqrt(val_loss)
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Time: {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSLE: {val_rmse:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'model_checkpoint_14.pth')
            
    print(f"Training Complete. Best Val RMSLE: {np.sqrt(best_val_loss):.4f}")
    
    # 3. Prediction (TTA)
    print("Generating Predictions (with TTA)...")
    model.load_state_dict(best_model_wts)
    model.eval()
    
    test_dataset = ConstructionDataset(pseudo_df, img_source, transform=val_transform) 
    # Just reusing validation transform for base
    
    predictions = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            if i % 100 == 0: print(f"  Predicting {i}/{len(test_dataset)}")
            
            # Manual TTA Loop because it's safer for image-by-image rotation
            # Load raw image again essentially or use dataset access
            # For speed, we will assume standard transform + random horiz flip trick or just rotation
            
            # Let's do a cleaner TTA using the loaded image in dataset
            # We need to access the underlying image before transform... 
            # Actually, standard TTA:
            # 1. Base
            # 2. HFlip
            # 3. VFlip
            # 4. Rotate 90
            
            # Since Dataset returns Tensor, we can just use Torch transforms on the tensor if we are careful,
            # but rotation is best done on PIL.
            # Simpler: Just train loop style inference.
            
            # To be consistent with Method 08, we should rotate 0,90,180,270.
            # But here we will stick to a simpler TTA: Base + HFlip to save inference time/complexity in this script
            # Or better: Just Base prediction if we trust the network.
            # Actually, User wants MAXIMUM performance. I should implement TTA.
            pass

    # Re-Using the efficient batch inference from predict_10_dl.py logic but adapted here
    # Actually, let's just create a test_loader and predict "Base" first to see if it works.
    # TTA is implemented in `predict_10_dl.py`. I'll implement a simple one here.
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    final_preds = []
    
    for images, _ in test_loader:
        images = images.to(device)
        
        # TTA: Simple 2x (Original + Horizontal Flip)
        # 1. Original
        out1 = model(images)
        
        # 2. Flip
        images_flip = torch.flip(images, [3]) # Flip width
        out2 = model(images_flip)
        
        avg_out = (out1 + out2) / 2
        final_preds.extend(avg_out.cpu().numpy().flatten())
        
    final_preds = np.expm1(final_preds)
    
    sub = pd.DataFrame()
    if 'data_id' in pseudo_df.columns:
        sub['data_id'] = pseudo_df['data_id']
    sub['construction_cost_per_m2_usd'] = final_preds
    sub.to_csv('submission014.csv', index=False)
    print("Submission saved to submission014.csv")

if __name__ == "__main__":
    train_and_predict()
