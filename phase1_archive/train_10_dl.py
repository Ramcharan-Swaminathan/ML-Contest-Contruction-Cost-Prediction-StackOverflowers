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

# Device configuration - User explicitly mentioned CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Config ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 150
IMAGE_SIZE = 224
DATA_PATH = 'dataset/train_tabular.csv'
IMAGE_DIR = 'dataset/train_composite'

# --- Dataset ---
class ConstructionDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_train=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Target
        target = np.log1p(row['construction_cost_per_m2_usd'])
        
        # Image Loading
        s2_filename = row.get('sentinel2_tiff_file_name')
        s2_path = os.path.join(self.image_dir, s2_filename) if isinstance(s2_filename, str) else None
        
        image = None
        if s2_path and os.path.exists(s2_path):
            try:
                with rasterio.open(s2_path) as src:
                    # Read RGB bands (4,3,2)
                    r = src.read(4) 
                    g = src.read(3) 
                    b = src.read(2) 
                    
                    def normalize_band(band):
                        band = band.astype(float)
                        p98 = np.percentile(band, 98) 
                        if p98 > 0:
                            band = np.clip(band, 0, p98) / p98
                        else:
                            band = np.clip(band, 0, 1)
                        return (band * 255).astype(np.uint8)
                    
                    rgb = np.dstack((normalize_band(r), normalize_band(g), normalize_band(b)))
                    image = Image.fromarray(rgb)
            except Exception as e:
                # print(f"Error loading {s2_path}: {e}")
                pass
        
        # Fallback for missing images
        if image is None:
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(0,0,0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(target, dtype=torch.float32)

# --- Model ---
class ConstructionNet(nn.Module):
    def __init__(self):
        super(ConstructionNet, self).__init__()
        # Load Pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)
        
        # Unfreeze all layers (End-to-End Fine-Tuning)
        for param in self.resnet.parameters():
            param.requires_grad = True
            
        # Replace Classification Head with Regression Head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.resnet(x)

def train(args):
    # 1. Prepare Data
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    
    # Train/Validation Split (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15), # Slight rotation augmentation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ConstructionDataset(train_df, IMAGE_DIR, transform=train_transform, is_train=True)
    val_dataset = ConstructionDataset(val_df, IMAGE_DIR, transform=val_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Setup Model
    print("Initializing Model...")
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
            targets = targets.to(device).unsqueeze(1) # [batch, 1]
            
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
        val_rmse_sum = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                # RMSE calculation for display (sqrt(mse))
                val_rmse_sum += torch.sum(torch.sqrt((outputs - targets)**2)).item()
        
        val_loss = val_loss / len(val_dataset)
        val_rmse = np.sqrt(val_loss) # Approximation
        
        # Scheduler Step
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val RMSLE: {val_rmse:.4f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'model_checkpoint_10.pth')
            # print("  --> Best Model Saved")
            
    print(f"Training Complete. Best Val RMSLE: {np.sqrt(best_val_loss):.4f}")

if __name__ == "__main__":
    train(None)
