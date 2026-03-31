import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import json
import os
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

# --- Config ---
TRAIN_TAB = 'dataset/train_tabular.csv'
TEST_TAB = 'evaluation_dataset/evaluation_tabular_no_target.csv'
TRAIN_FEATS = 'dataset/image_features_train.csv'
TEST_FEATS = 'evaluation_dataset/image_features_test.csv'
DOMAIN_LIST = 'domain_features.json'
SUBMISSION_FILE = 'submission035.csv'
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
        
    # Load Domain Features
    with open(DOMAIN_LIST, 'r') as f:
        features = json.load(f)
        
    print(f"Using {len(features)} Domain Features: {features}")
    
    # Prep X/y
    y = np.log1p(df_train['construction_cost_per_m2_usd'])
    groups = df_train['geolocation_name']
    
    # Align Train/Test columns
    X_train = df_train[features].copy()
    X_test = df_test[features].copy()
    
    return X_train, y, groups, X_test

def objective(trial, X, y, groups):
    # Search Space - Adjusted for Small Feature Set (6 features)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': SEED,
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 8, 32), # Low depth allowed
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
    }
    
    scores = []
    gkf = GroupKFold(n_splits=5)
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=callbacks)
        
        preds = model.predict(X_val)
        rmse = np.sqrt(np.mean((preds - y_val)**2))
        scores.append(rmse)
        
    return np.mean(scores)

def train_final_model(best_params, X, y, groups, X_test):
    print("Training Final Model with Best Params:", best_params)
    
    gkf = GroupKFold(n_splits=5)
    test_preds_accum = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        params = best_params.copy()
        params['n_estimators'] = 2000 
        params['objective'] = 'regression'
        params['metric'] = 'rmse'
        params['random_state'] = SEED + fold
        
        model = lgb.LGBMRegressor(**params)
        callbacks = [lgb.early_stopping(stopping_rounds=100)]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=callbacks)
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        
        # Test Preds
        test_preds_accum += model.predict(X_test)
        
        rmse = np.sqrt(np.mean((val_pred - y_val)**2))
        print(f"Fold {fold} RMSE: {rmse:.4f}")

    avg_test_preds = test_preds_accum / 5.0
    final_rmse = np.sqrt(np.mean((oof_preds - y)**2))
    print(f"Overall CV RMSLE: {final_rmse:.5f}")
    
    return np.expm1(avg_test_preds)

if __name__ == "__main__":
    X, y, groups, X_test = load_data()
    
    print("Starting Optuna Tuning...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y, groups), n_trials=N_TRIALS)
    
    print("Best Params:", study.best_params)
    
    final_preds = train_final_model(study.best_params, X, y, groups, X_test)
    
    # Save
    df_test = pd.read_csv(TEST_TAB)
    sub = pd.DataFrame({'data_id': df_test['data_id'], 'construction_cost_per_m2_usd': final_preds})
    sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved {SUBMISSION_FILE}")
