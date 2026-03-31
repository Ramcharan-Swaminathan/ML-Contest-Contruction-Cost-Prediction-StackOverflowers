import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

# --- Config ---
TRAIN_DATA = 'dataset/train_tabular.csv'
TEST_DATA = 'evaluation_tabular_no_target.csv'
TRAIN_IMG_FEATS = 'dataset/image_features_train.csv'
TEST_IMG_FEATS = 'evaluation_dataset/image_features_test.csv'
PSEUDO_LABEL_FILE = 'submission_final.csv' 

# --- Best Hyperparameters (M21 Base) ---
# We use M21 params as a strong baseline
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
    'colsample_bytree': 0.8
}

def load_and_merge():
    print("Loading Tabular Data...")
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    print("Loading Image Features...")
    if os.path.exists(TRAIN_IMG_FEATS):
        train_imgs = pd.read_csv(TRAIN_IMG_FEATS)
        # Merge on data_id
        train_df = train_df.merge(train_imgs, on='data_id', how='left')
    else:
        print("Warning: Train Image Features not found!")

    if os.path.exists(TEST_IMG_FEATS):
        test_imgs = pd.read_csv(TEST_IMG_FEATS)
        test_df = test_df.merge(test_imgs, on='data_id', how='left')
    else:
         print("Warning: Test Image Features not found!")
         
    return train_df, test_df

def feature_engineering(df):
    # --- Research Features ---
    # 1. Remote Island (>800km)
    if 'straight_distance_to_capital_km' in df.columns:
        df['is_remote_island'] = (df['straight_distance_to_capital_km'] > 800).astype(int)
        
    # 2. Risk Index
    if 'flood_risk' in df.columns and 'seismic_risk' in df.columns:
        # Fill NA with median or 0? Paper says only 4 missing.
        fr = df['flood_risk'].fillna(df['flood_risk'].median())
        sr = df['seismic_risk'].fillna(df['seismic_risk'].median())
        df['risk_index'] = fr + sr
        
    # --- Date Components ---
    if 'quarter_label' in df.columns:
        df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-Q')[1]) if '-Q' in str(x) else 0)
        df['year'] = df['quarter_label'].apply(lambda x: int(x.split('-')[0]) if '-' in str(x) else 2020)
        
    return df

def add_spatial_encoding(train_df, test_df, pseudo_df=None):
    # Method 20 Logic (Standard KFold for generating the feature values)
    print("Adding Spatial Target Encodings...")
    target_cols = ['geolocation_name', 'country', 'region_economic_classification']
    train_df['log_target'] = np.log1p(train_df['construction_cost_per_m2_usd'])
    
    # We use standard KFold to generate the OOF features for training
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for col in target_cols:
        train_df[f'{col}_target_enc'] = 0.0
        for tr_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
            global_mean = X_tr['log_target'].mean()
            mapping = X_tr.groupby(col)['log_target'].mean()
            train_df.loc[val_idx, f'{col}_target_enc'] = X_val[col].map(mapping).fillna(global_mean)
            
    mappings = {}
    global_mean = train_df['log_target'].mean()
    for col in target_cols:
        mappings[col] = train_df.groupby(col)['log_target'].mean()
        test_df[f'{col}_target_enc'] = test_df[col].map(mappings[col]).fillna(global_mean)
        if pseudo_df is not None:
             pseudo_df[f'{col}_target_enc'] = pseudo_df[col].map(mappings[col]).fillna(global_mean)
             
    train_df = train_df.drop(columns=['log_target'])
    return train_df, test_df, pseudo_df

def train_m24():
    train_df, test_df = load_and_merge()
    
    # Load Pseudo
    pseudo_df = pd.read_csv(PSEUDO_LABEL_FILE)
    if 'data_id' in test_df.columns and 'data_id' in pseudo_df.columns:
        pseudo_df_full = test_df.copy()
        label_map = dict(zip(pseudo_df['data_id'], pseudo_df['construction_cost_per_m2_usd']))
        pseudo_df_full['construction_cost_per_m2_usd'] = pseudo_df_full['data_id'].map(label_map)
    else:
        pseudo_df_full = test_df.copy()
        pseudo_df_full['construction_cost_per_m2_usd'] = pseudo_df['construction_cost_per_m2_usd']
        
    # Feature Engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    pseudo_df_full = feature_engineering(pseudo_df_full)
    
    # Spatial Encoding
    train_df, test_df, pseudo_df_full = add_spatial_encoding(train_df, test_df, pseudo_df_full)
    
    # Prep Categoricals
    for df in [train_df, test_df, pseudo_df_full]:
         for col in df.select_dtypes(include=['object']).columns:
            if col not in ['sentinel2_tiff_file_name', 'viirs_tiff_file_name']:
                df[col] = df[col].astype('category')
                
    target_col = 'construction_cost_per_m2_usd'
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'quarter_label', target_col]
    features = [c for c in train_df.columns if c not in drop_cols]
    
    print(f"Total Features: {len(features)}")
    
    X = train_df[features]
    y = np.log1p(train_df[target_col])
    groups = train_df['geolocation_name'] # For GroupKFold
    
    X_pseudo = pseudo_df_full[features]
    y_pseudo = np.log1p(pseudo_df_full[target_col])
    
    # --- GroupKFold Validation (Research Recommendation) ---
    # Note: We validate on Train only using GroupKFold to check robustness.
    # For final training, we usually use all data. 
    # But let's check CV score first using GroupKFold.
    
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(X))
    
    print("Starting GroupKFold Cross-Validation...")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Add Pseudo Data to Training Fold
        X_tr_fold = pd.concat([X_tr, X_pseudo], axis=0)
        y_tr_fold = pd.concat([y_tr, y_pseudo], axis=0)
        
        train_data = lgb.Dataset(X_tr_fold, label=y_tr_fold)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        bst = lgb.train(params, train_data, num_boost_round=5000, 
                        valid_sets=[val_data], 
                        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)])
                        
        oof_preds[val_idx] = bst.predict(X_val)
        
    score = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"GroupKFold RMSLE Score: {score:.5f}")
    
    # --- Final Training (Full Data) ---
    print("Retraining on Full Data + Pseudo...")
    X_full = pd.concat([X, X_pseudo], axis=0)
    y_full = pd.concat([y, y_pseudo], axis=0)
    
    train_data = lgb.Dataset(X_full, label=y_full)
    bst = lgb.train(params, train_data, num_boost_round=1200) # Slightly more than avg optimal
    
    preds_log = bst.predict(test_df[features])
    preds = np.expm1(preds_log)
    
    sub = pd.DataFrame()
    if 'data_id' in test_df.columns:
        sub['data_id'] = test_df['data_id']
    sub['construction_cost_per_m2_usd'] = preds
    sub.to_csv('submission024.csv', index=False)
    print("Done. Saved submission024.csv")

if __name__ == "__main__":
    train_m24()
