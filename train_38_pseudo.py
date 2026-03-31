import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import json
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# --- Config ---
TRAIN_TAB = 'dataset/train_tabular.csv'
TEST_TAB = 'evaluation_dataset/evaluation_tabular_no_target.csv'
TRAIN_FEATS = 'dataset/image_features_train.csv'
TEST_FEATS = 'evaluation_dataset/image_features_test.csv'
PSEUDO_LABEL_FILE = 'phase1_archive/sub_ens_kestav.csv'
DOMAIN_LIST = 'domain_features.json'
SUBMISSION_FILE = 'submission038.csv'
N_TRIALS = 20
SEED = 42

def load_data():
    print("Loading Data...")
    df_train = pd.read_csv(TRAIN_TAB)
    df_test = pd.read_csv(TEST_TAB)
    
    # Merge Image Features
    if os.path.exists(TRAIN_FEATS) and os.path.exists(TEST_FEATS):
        f_train = pd.read_csv(TRAIN_FEATS)
        f_test = pd.read_csv(TEST_FEATS)
        df_train = df_train.merge(f_train, on='data_id', how='left')
        df_test = df_test.merge(f_test, on='data_id', how='left')
        
    # Load Pseudo Labels
    print(f"Loading Pseudo Labels from {PSEUDO_LABEL_FILE}...")
    df_pseudo = pd.read_csv(PSEUDO_LABEL_FILE)
    
    # Merge Pseudo Labels into Test
    # Ensure alignment
    df_test = df_test.merge(df_pseudo, on='data_id', how='left')
    
    # Combine Train and Pseudo-Test
    # Train: 'construction_cost_per_m2_usd' comes from CSV
    # Test: 'construction_cost_per_m2_usd' comes from PSEUDO_LABEL_FILE
    
    df_combined = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    print(f"Combined Data Size: {len(df_combined)} (Train: {len(df_train)}, Test: {len(df_test)})")
        
    # Load Domain Features
    with open(DOMAIN_LIST, 'r') as f:
        features = json.load(f)
        
    print(f"Using {len(features)} Domain Features")
    
    # Prep X/y
    y_combined = np.log1p(df_combined['construction_cost_per_m2_usd'])
    X_combined = df_combined[features].copy()
    
    # For Final Prediction on Test, we just predict on the Test portion again (Refinement)
    X_test_final = df_test[features].copy()
            
    return X_combined, y_combined, X_test_final

def objective(trial, X, y):
    # K-Fold instead of GroupKFold because we mixed Train/Test and Test has no 'geolocation_name' usually (or we ignore groups for pseudo)
    # Actually, Test dataset does not have 'geolocation_name' in current loading? 
    # Check if 'geolocation_name' is in features. No, it's not.
    # GroupKFold requires groups. Let's use KFold (random shuffle) effectively treating pseudo-labels as high confidence.
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 8, 32),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
    }
    
    scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=callbacks)
        
        preds = model.predict(X_val)
        rmse = np.sqrt(np.mean((preds - y_val)**2))
        scores.append(rmse)
        
    return np.mean(scores)

def train_final_model(best_params, X, y, X_test_final):
    print("Training Final Model on Combined Data...")
    
    # Train on ALL data (Train + Pseudo)
    params = best_params.copy()
    params['n_estimators'] = 2000 
    params['objective'] = 'regression'
    params['metric'] = 'rmse'
    params['random_state'] = SEED
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, eval_metric='rmse')
    
    # Predict on Test (Refining the pseudo labels)
    final_preds = model.predict(X_test_final)
    
    return np.expm1(final_preds)

if __name__ == "__main__":
    X_combined, y_combined, X_test_final = load_data()
    
    print("Starting Optuna Tuning on Expanded Dataset...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_combined, y_combined), n_trials=N_TRIALS)
    
    print("Best Params:", study.best_params)
    
    final_preds = train_final_model(study.best_params, X_combined, y_combined, X_test_final)
    
    # Save
    df_test_proto = pd.read_csv(TEST_TAB)
    sub = pd.DataFrame({'data_id': df_test_proto['data_id'], 'construction_cost_per_m2_usd': final_preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved {SUBMISSION_FILE}")
