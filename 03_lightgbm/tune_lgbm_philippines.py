import pandas as pd
import numpy as np
import os
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import logging

# Enable Optuna logs
optuna.logging.set_verbosity(optuna.logging.INFO)

def tune_lgbm():
    # Verify GPU
    try:
        print("\nVerifying LightGBM GPU support...")
        from lightgbm import train as lgb_train, Dataset as lgb_Dataset
        # Dummy train
        d = lgb_Dataset(np.array([[1,2],[2,3]]), label=np.array([0,1]))
        p = {'device': 'cuda', 'verbosity': -1, 'min_data_in_leaf': 1}
        lgb_train(p, d, num_boost_round=1)
        print("  [SUCCESS] LightGBM initialized with device='cuda' without error.")
    except Exception as e:
        print(f"  [WARNING] LightGBM failed to use GPU: {e}")
        print("  Please ensure lightgbm is installed with GPU support (pip install check).")

    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_CLEAN_DIR = os.path.join(PROJECT_ROOT, "data_clean")
    JAPAN_PATH = os.path.join(DATA_CLEAN_DIR, "train_japan.csv")
    PHILIPPINES_PATH = os.path.join(DATA_CLEAN_DIR, "train_philippines.csv")
    
    # Load Data
    print("Loading data...")
    df_jp = pd.read_csv(JAPAN_PATH)
    df_ph = pd.read_csv(PHILIPPINES_PATH)
    
    datasets = [('Philippines', df_ph)]
    
    best_params_dict = {}

    for name, df in datasets:
        print(f"\n--- Tuning LightGBM for {name} ---")
        
        target_col = 'construction_cost_per_m2_usd'
        y = df[target_col]
        X = df.drop(columns=[target_col])
        y_log = np.log1p(y)
        
        # Identify Categoricals for LightGBM
        # We need to convert object columns to 'category' dtype
        # Note: We already target encoded 'geolocation_name' in cleaning? 
        # Let's check cleaning script. Yes, 'geolocation_name' became float.
        # But other cols might be categorical strings: 'quarter_label', 'region', etc.
        
        obj_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')
            
        def objective(trial):
            param = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'n_jobs': -1,
                'boosting_type': 'gbdt',
                'device': 'cuda',
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_estimators': 1000, 
            }
            
            print(f"  [Trial {trial.number}] Params: device={param['device']}, learning_rate={param['learning_rate']:.4f} ...")
            
            # 5-Fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rmses = []
            
            # Manual CV loop to use LightGBM callbacks if needed (though sklearn API handles it too)
            # Using simple k-fold with sklearn API
            # Note: LGBMRegressor doesn't take 'callbacks' in constructor easily for early stopping in CV without fit params
            # We will use simple fixed n_estimators with slightly conservative LR, or rely on pruning.
            
            # Actually, let's use lgb.cv for efficiency? 
            # But lgb.cv returns history.
            # Let's stick to manual fold loop to stay consistent with metric calculation
            
            for train_idx, val_idx in kf.split(X, y_log):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
                
                dtrain = lgb.Dataset(X_train, label=y_train)
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                
                # Suppress output
                bst = lgb.train(
                    param, dtrain, 
                    valid_sets=[dval], 
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                
                preds = bst.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                rmses.append(rmse)
            
            return np.mean(rmses)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50) # Run 20 trials for speed
        
        print(f"  Best RMSLE: {study.best_value:.4f}")
        print("  Best Params:")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")
        
        best_params_dict[name] = study.best_params

    return best_params_dict

if __name__ == "__main__":
    tune_lgbm()
