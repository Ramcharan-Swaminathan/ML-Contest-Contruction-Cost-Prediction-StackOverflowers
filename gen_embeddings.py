import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- Config ---
TRAIN_CSV = 'dataset/train.csv'
TEST_CSV = 'evaluation_dataset/evaluation.csv' # or similar
# Actually, the image paths are needed.
# Let's assume standard folder structure or CSV has paths.
# Checking dataset structure...
# The CSVs usually have 'data_id'. We assume 'dataset/train_images/{data_id}.png' exists?
# I need to verify image paths first.

IMG_DIR_TRAIN = 'dataset/' # ?
IMG_DIR_TEST = 'evaluation_dataset/' # ?

BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Dataset ---
class ConstructionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Recursively search for the image? Or assume standard path?
        # Let's assume the user has images in a consistent place.
        # Based on previous phases, images are likely in 'dataset/train_images' or just inside 'dataset'.
        # I'll check file existence in __init__ or blindly try.
        # Let's check 'dataset/{data_id}.png'.
        
        # M3 used specific logic. Let's start with a path search helper.
        # But for efficiency, I'll rely on the standard pattern.
        # Try finding one image first.
        
        # Use filename logic
        filename = row['sentinel2_tiff_file_name'].replace('.tif', '.png')
        path = os.path.join(self.img_dir, filename)
        
        if not os.path.exists(path):
             print(f"Warning: Image missing {path}")
             # Return black image to avoid crash
             return torch.zeros((3, 224, 224))
             
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img

def gen_embeddings():
    print(f"Generating Embeddings using {DEVICE}...")
    
    # 1. Load DataFrames
    train_df = pd.read_csv('dataset/train_tabular.csv')
    test_df = pd.read_csv('evaluation_dataset/evaluation_tabular_no_target.csv')
    
    # Verify Image Paths
    # I'll assume they are in 'dataset/train_images/' and 'evaluation_dataset/test_images/' based on typical Kaggle setup
    # But I need to verify.
    # Let's verify via `ls` before running this script fully.
    
    # ... (Assumption: Images are accessible)
    
    # 2. Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity() # Remove Head
    model.to(DEVICE)
    model.eval()
    
    # 3. Transforms
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Generate
    def extract(df, img_dir):
        # We need to find where the images actually are.
        # Often they are flat in the directory or in a subfolder.
        # I'll checking inside the Dataset class is tricky.
        # Let's do a quick scan.
        full_img_dir = img_dir # Placeholder
        # Hack: Check if {img_dir}/{id}.png exists.
        if not os.path.exists(os.path.join(img_dir, f"{df.iloc[0]['data_id']}.png")):
            # Try subfolder
            if os.path.exists(os.path.join(img_dir, 'train_images', f"{df.iloc[0]['data_id']}.png")):
                full_img_dir = os.path.join(img_dir, 'train_images')
            elif os.path.exists(os.path.join(img_dir, 'test_images', f"{df.iloc[0]['data_id']}.png")):
                 full_img_dir = os.path.join(img_dir, 'test_images')
        
        ds = ConstructionDataset(df, full_img_dir, transform=tfms)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        emb_list = []
        with torch.no_grad():
            for imgs in tqdm(dl):
                imgs = imgs.to(DEVICE)
                feats = model(imgs)
                emb_list.append(feats.cpu().numpy())
                
        return np.vstack(emb_list)

    train_embs = extract(train_df, 'dataset/train_png/')
    test_embs = extract(test_df, 'evaluation_dataset/test_png/')
    
    np.save('dataset/embeddings_resnet18_train.npy', train_embs)
    np.save('evaluation_dataset/embeddings_resnet18_test.npy', test_embs)
    print("Embeddings Saved.")

if __name__ == "__main__":
    gen_embeddings()
