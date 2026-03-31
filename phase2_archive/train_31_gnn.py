import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import os

# --- Config ---
TRAIN_TAB = 'dataset/train_tabular.csv'
TEST_TAB = 'evaluation_dataset/evaluation_tabular_no_target.csv'
TRAIN_FEATS = 'dataset/image_features_train.csv'
TEST_FEATS = 'evaluation_dataset/image_features_test.csv'
SUBMISSION_FILE = 'submission031.csv'

# Hyperparams
K_NEIGHBORS = 15 # Spatial context size
HIDDEN_DIM = 64
DROPOUT = 0.2
LR = 0.01
EPOCHS = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_prepare_data():
    print("Loading Data...")
    # Load Tabular
    df_train = pd.read_csv(TRAIN_TAB)
    df_test = pd.read_csv(TEST_TAB)
    
    # Load Image Features
    if os.path.exists(TRAIN_FEATS) and os.path.exists(TEST_FEATS):
        print("Merging Image Features...")
        f_train = pd.read_csv(TRAIN_FEATS)
        f_test = pd.read_csv(TEST_FEATS)
        df_train = df_train.merge(f_train, on='data_id', how='left')
        df_test = df_test.merge(f_test, on='data_id', how='left')
    
    # Identify Split
    df_train['is_train'] = True
    df_test['is_train'] = False
    df_test['construction_cost_per_m2_usd'] = 0 # Dummy valid
    
    # Concat
    # Align columns
    cols = [c for c in df_train.columns if c in df_test.columns]
    full_df = pd.concat([df_train[cols], df_test[cols]], axis=0, ignore_index=True)
    
    return full_df, df_train.shape[0]

def build_graph(df):
    print(f"Building Visual Graph (K={K_NEIGHBORS})...")
    # visual features for KNN
    # We use the extracted image statistics as the "coordinates" for visual similarity
    visual_cols = [c for c in df.columns if (c.startswith('s2_') or c.startswith('viirs_')) and not c.endswith('_name')]
    coords = df[visual_cols].values
    
    # Normalize coords for KNN
    coords = StandardScaler().fit_transform(coords)
    coords = np.nan_to_num(coords)
    
    # KNN
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Edges (Source -> Target)
    # indices[:, 0] is self, so start from 1
    edge_index = []
    for i in range(indices.shape[0]):
        for j in indices[i, 1:]: # Skip self
            edge_index.append([i, j])
            edge_index.append([j, i]) # Undirected
            
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print(f"Graph Created: {df.shape[0]} nodes, {edge_index.shape[1]} edges.")
    return edge_index

def prepare_features(df):
    # Select Features (Numerical Only)
    drop = ['data_id', 'is_train', 'construction_cost_per_m2_usd', 'quarter_label', 
            'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'geolocation_name', 
            'country', 'region_economic_classification', 'seismic_hazard_zone', 
            'tropical_cyclone_wind_risk', 'koppen_climate_zone']
    
    feats = df.drop(columns=drop, errors='ignore')
    
    # One-Hot Encode Categoricals if any remain (none above, but just in case)
    feats = pd.get_dummies(feats, dummy_na=True)
    feats = feats.fillna(0)
    
    # Scale
    scaler = StandardScaler()
    x = scaler.fit_transform(feats)
    return torch.tensor(x, dtype=torch.float)

class SpatialSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        
        return self.head(x)

def train_gnn():
    full_df, n_train = load_and_prepare_data()
    edge_index = build_graph(full_df).to(DEVICE)
    x = prepare_features(full_df).to(DEVICE)
    
    # Targets
    y_raw = full_df['construction_cost_per_m2_usd'].values
    y_log = np.log1p(y_raw)
    y = torch.tensor(y_log, dtype=torch.float).view(-1, 1).to(DEVICE)
    
    # Indices
    train_indices = np.where(full_df['is_train'])[0]
    test_indices = np.where(~full_df['is_train'])[0]
    
    # CV
    gkf = GroupKFold(n_splits=5)
    groups = full_df.loc[train_indices, 'geolocation_name']
    
    oof_preds = np.zeros(n_train)
    test_preds_accum = np.zeros(len(test_indices))
    
    print(f"Starting Training on {DEVICE}...")
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_indices, y_log[train_indices], groups)):
        # Map back to global indices
        tr_global = train_indices[tr_idx]
        val_global = train_indices[val_idx]
        
        model = SpatialSAGE(x.shape[1], HIDDEN_DIM).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = criterion(out[tr_global], y[tr_global])
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(out[val_global], y[val_global]).item()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
        
        print(f"Fold {fold} Best Val MSE: {best_val_loss:.4f} (RMSLE: {np.sqrt(best_val_loss):.4f})")
        
        # Predict on Val and Test
        model.eval()
        with torch.no_grad():
            final_out = model(x, edge_index)
            oof_preds[val_idx] = final_out[val_global].cpu().numpy().flatten()
            test_preds_accum += final_out[test_indices].cpu().numpy().flatten()

    # Scores
    rmsle = np.sqrt(np.mean((oof_preds - y_log[train_indices].flatten())**2))
    print(f"overall CV RMSLE: {rmsle:.5f}")
    
    # Final Submission (Average of Fold Models)
    avg_test_preds_log = test_preds_accum / 5
    avg_test_preds = np.expm1(avg_test_preds_log)
    
    sub = pd.DataFrame({
        'data_id': full_df.loc[test_indices, 'data_id'],
        'construction_cost_per_m2_usd': avg_test_preds
    })
    sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved {SUBMISSION_FILE}")

if __name__ == "__main__":
    train_gnn()
