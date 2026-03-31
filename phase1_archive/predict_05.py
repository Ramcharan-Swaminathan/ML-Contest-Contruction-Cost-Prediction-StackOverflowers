import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import argparse
import rasterio
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
import pickle
import warnings

warnings.filterwarnings('ignore')

EXTRACTION_DEVICE = torch.device('cpu') 

def get_resnet_embedder():
    print("Loading pretrained ResNet18...")
    resnet = models.resnet18(pretrained=True)
    modules = list(resnet.children())[:-1]
    embedder = torch.nn.Sequential(*modules)
    embedder.to(EXTRACTION_DEVICE)
    embedder.eval()
    return embedder

def extract_cnn_embeddings(df, image_dir):
    print("Extracting CNN embeddings for prediction...")
    
    embedder = get_resnet_embedder()
    
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    embeddings = []
    
    total = len(df)
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"Processing row {idx}/{total}")
            
        s2_filename = row.get('sentinel2_tiff_file_name')
        s2_path = os.path.join(image_dir, s2_filename) if isinstance(s2_filename, str) else None
        
        emb_vector = None
        if s2_path and os.path.exists(s2_path):
            try:
                with rasterio.open(s2_path) as src:
                    r = src.read(4) 
                    g = src.read(3) 
                    b = src.read(2) 
                    
                    def normalize(band):
                        band = band.astype(float)
                        p98 = np.percentile(band, 98)
                        if p98 > 0:
                            band = np.clip(band, 0, p98) / p98
                        else:
                            band = np.clip(band, 0, 1)
                        return (band * 255).astype(np.uint8)
                    
                    rgb = np.dstack((normalize(r), normalize(g), normalize(b)))
                    
                    pil_img = Image.fromarray(rgb)
                    input_tensor = preprocess(pil_img)
                    input_batch = input_tensor.unsqueeze(0).to(EXTRACTION_DEVICE)
                    
                    with torch.no_grad():
                        output = embedder(input_batch)
                        emb_vector = output.cpu().numpy().flatten()
                        
            except Exception:
                pass
        
        if emb_vector is None:
            emb_vector = np.zeros(512)
            
        embeddings.append(emb_vector)
        
    embeddings = np.array(embeddings)
    
    # Load PCA model
    if not os.path.exists('pca_model_05.pkl'):
        raise FileNotFoundError("PCA model not found! Run train_05.py first.")
        
    with open('pca_model_05.pkl', 'rb') as f:
        pca = pickle.load(f)
        
    print(f"Applying PCA (n={pca.n_components_})...")
    pca_embeddings = pca.transform(embeddings)
    
    # Add to DF
    n_components = pca.n_components_
    emb_cols = [f'emb_{i}' for i in range(n_components)]
    emb_df = pd.DataFrame(pca_embeddings, columns=emb_cols, index=df.index)
    
    return pd.concat([df, emb_df], axis=1)

def predict(args):
    # Args
    if isinstance(args, argparse.Namespace):
        model_path = args.model_path 
        data_path = args.data_path
        output_path = args.output_path
        image_dir = args.image_dir
    else:
         model_path = 'model_checkpoint_05.txt'
         data_path = 'evaluation_tabular_no_target.csv'
         output_path = 'submission005.csv'
         image_dir = 'evaluation_dataset/evaluation_composite'

    if data_path is None:
         data_path = 'evaluation_tabular_no_target.csv'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data not found at {data_path}")

    print(f"Loading model from {model_path}...")
    bst = lgb.Booster(model_file=model_path)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    ids = df['data_id'] if 'data_id' in df.columns else None

    # 1. Embeddings
    df = extract_cnn_embeddings(df, image_dir)
    
    # 2. Drop columns
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', 'construction_cost_per_m2_usd']
    
    if 'quarter_label' in df.columns:
        df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if isinstance(x, str) and '-Q' in x else 0)
    
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # 3. Categoricals
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
        
    # 4. Feature Matching
    model_features = bst.feature_name()
    missing = [f for f in model_features if f not in X.columns]
    
    if missing:
        print(f"WARNING: Input data missing features: {missing}. Filling with NaN.")
        for f in missing:
            X[f] = np.nan
            
    X = X[model_features]
    print("Features matched.")
    
    preds_log = bst.predict(X)
    preds_actual = np.expm1(preds_log)
    
    submission = pd.DataFrame()
    if ids is not None:
        submission['data_id'] = ids
        
    submission['construction_cost_per_m2_usd'] = preds_actual
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str, default='model_checkpoint_05.txt')
    parser.add_argument('--output_path', type=str, default='submission005.csv')
    parser.add_argument('--image_dir', type=str, default='evaluation_dataset/evaluation_composite')
    
    args = parser.parse_args()
    predict(args)
