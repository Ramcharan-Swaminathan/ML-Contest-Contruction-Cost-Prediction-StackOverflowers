import pandas as pd
import numpy as np
import os
import argparse
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Config ---
IMAGE_SIZE = 224
BATCH_SIZE = 32

# --- Model ---
class ConstructionNet(nn.Module):
    def __init__(self):
        super(ConstructionNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False) # Structure only
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.resnet(x)

# --- Dataset for TTA ---
class ConstructionTestDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s2_filename = row.get('sentinel2_tiff_file_name')
        s2_path = os.path.join(self.image_dir, s2_filename) if isinstance(s2_filename, str) else None
        
        image = None
        if s2_path and os.path.exists(s2_path):
            try:
                with rasterio.open(s2_path) as src:
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
            except Exception:
                pass
        
        if image is None:
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(0,0,0))
            
        # TTA: Return 4 images (0, 90, 180, 270) stacked
        rotations = [0, 90, 180, 270]
        tensors = []
        
        for angle in rotations:
            if angle == 0:
                img_t = image
            else:
                img_t = image.rotate(angle)
            
            if self.transform:
                tensors.append(self.transform(img_t))
        
        return torch.stack(tensors) # Shape: [4, 3, 224, 224]

def predict(args):
    data_path = args.data_path
    output_path = args.output_path
    image_dir = args.image_dir
    model_path = args.model_path
    
    if data_path is None:
         data_path = 'evaluation_tabular_no_target.csv'
         
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ConstructionTestDataset(df, image_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Loading model from {model_path}...")
    model = ConstructionNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    predictions = []
    
    print("Starting Inference (Sampled TTA)...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 5 == 0:
                print(f"Batch {i}/{len(dataloader)}")
                
            # batch shape: [batch_size, 4, 3, 224, 224]
            # We need to flatten to [batch_size * 4, 3, 224, 224]
            bs, n_aug, c, h, w = batch.size()
            inputs = batch.view(-1, c, h, w).to(device)
            
            outputs = model(inputs) # [batch_size * 4, 1]
            outputs = outputs.view(bs, n_aug) # [batch_size, 4]
            
            # Average predictions in log space
            avg_log_preds = outputs.mean(dim=1).cpu().numpy()
            
            # Convert to actual cost
            pred_costs = np.expm1(avg_log_preds)
            predictions.extend(pred_costs)
            
    # Save
    submission = pd.DataFrame()
    if 'data_id' in df.columns:
        submission['data_id'] = df['data_id']
    submission['construction_cost_per_m2_usd'] = predictions
    
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str, default='submission010.csv')
    parser.add_argument('--model_path', type=str, default='model_checkpoint_10.pth')
    parser.add_argument('--image_dir', type=str, default='evaluation_dataset/evaluation_composite')
    
    args = parser.parse_args()
    predict(args)
