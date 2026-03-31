import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --- Config ---
TRAIN_TAB = 'dataset/train_tabular.csv'
TEST_TAB = 'evaluation_dataset/evaluation_tabular_no_target.csv'
TRAIN_FEATS = 'dataset/image_features_train.csv'
TEST_FEATS = 'evaluation_dataset/image_features_test.csv'
DOMAIN_FEATS_FILE = 'domain_features.json' # M35
EVOLVED_FEATS_FILE = 'evolved_features.json' # M40
SEED = 42

def get_data():
    train = pd.read_csv(TRAIN_TAB)
    test = pd.read_csv(TEST_TAB)
    
    # Merge Stats
    if pd.read_csv(TRAIN_FEATS) is not None:
        ft = pd.read_csv(TRAIN_FEATS)
        ftest = pd.read_csv(TEST_FEATS)
        train = train.merge(ft, on='data_id', how='left')
        test = test.merge(ftest, on='data_id', how='left')
        
    y = np.log1p(train['construction_cost_per_m2_usd'])
    return train, test, y

def get_knn_oof(train, test, y):
    # M21 Logic: ResNet Embeddings (Need to load embeddings? assuming we don't have them easily loaded here, 
    # but we can use the 'image_features' as a proxy for M21-like behavior? 
    # No, M21 used 'embeddings_resnet18_train.npy'. I should use that if available.
    
    # Check for embeddings
    try:
        emb_train = np.load('dataset/embeddings_resnet18_train.npy')
        emb_test = np.load('evaluation_dataset/embeddings_resnet18_test.npy')
        print("Loaded ResNet Embeddings for M21 OOF.")
    except:
        print("Embeddings not found! Using standard scalar features for KNN proxy (Suboptimal).")
        # Fallback to scaled features
        cols = [c for c in train.columns if 'mean' in c or 'std' in c]
        emb_train = train[cols].fillna(0).values
        emb_test = test[cols].fillna(0).values
        scaler = StandardScaler()
        emb_train = scaler.fit_transform(emb_train)
        emb_test = scaler.transform(emb_test)

    # KNN CV
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for train_idx, val_idx in kf.split(emb_train):
        # Fit on Train Fold
        knn = NearestNeighbors(n_neighbors=50, metric='cosine')
        knn.fit(emb_train[train_idx])
        
        # Predict on Val Fold
        dists, idxs = knn.kneighbors(emb_train[val_idx])
        # Average target of neighbors
        # y is accessible via train_idx mapping?
        # We need y values of the neighbors.
        # idxs returns indices relative to 'emb_train[train_idx]'. No, it returns indices into the fitted data.
        # The fitted data is 'emb_train[train_idx]'.
        # So idxs[i, j] is the index in 'train_idx'.
        
        # Map localized indices back to global y
        global_neighbor_indices = train_idx[idxs] 
        neighbor_targets = y.iloc[global_neighbor_indices.flatten()].values.reshape(idxs.shape)
        
        # Average (M21 logic was arithmetic mean of log targets usually, or log of mean targets. Let's do mean of log targets for RMSLE)
        fold_preds = np.mean(neighbor_targets, axis=1)
        oof_preds[val_idx] = fold_preds
        
        # Predict on Test (Average over folds)
        dists_test, idxs_test = knn.kneighbors(emb_test)
        global_neighbor_indices_test = train_idx[idxs_test]
        neighbor_targets_test = y.iloc[global_neighbor_indices_test.flatten()].values.reshape(idxs_test.shape)
        fold_test_preds = np.mean(neighbor_targets_test, axis=1)
        test_preds += fold_test_preds / 5.0
        
    return oof_preds, test_preds

def get_lgbm_oof(train, test, y, feature_list, name):
    print(f"Generating OOF for {name}...")
    X = train[feature_list]
    X_test = test[feature_list]
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'seed': SEED
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for train_idx, val_idx in kf.split(X, y):
        model = lgb.LGBMRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx]) 
        oof_preds[val_idx] = model.predict(X.iloc[val_idx])
        test_preds += model.predict(X_test) / 5.0
        
    return oof_preds, test_preds

if __name__ == "__main__":
    train, test, y = get_data()
    
    # 1. M21 OOF
    m21_oof, m21_test = get_knn_oof(train, test, y)
    
    # 2. M35 OOF (Domain)
    with open(DOMAIN_FEATS_FILE, 'r') as f: m35_feats = json.load(f)
    m35_oof, m35_test = get_lgbm_oof(train, test, y, m35_feats, "M35")
    
    # 3. M40 OOF (Evolved)
    with open(EVOLVED_FEATS_FILE, 'r') as f: m40_feats = json.load(f)
    m40_oof, m40_test = get_lgbm_oof(train, test, y, m40_feats, "M40")
    
    # Save
    df_oof = pd.DataFrame({
        'data_id': train['data_id'],
        'target': y,
        'm21': m21_oof,
        'm35': m35_oof,
        'm40': m40_oof
    })
    df_oof.to_csv('oofs_train.csv', index=False)
    
    df_pred = pd.DataFrame({
        'data_id': test['data_id'],
        'm21': m21_test,
        'm35': m35_test,
        'm40': m40_test
    })
    df_pred.to_csv('oofs_test.csv', index=False)
    print("OOFs Generated.")
