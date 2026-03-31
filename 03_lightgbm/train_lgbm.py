import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_lgbm():
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_CLEAN_DIR = os.path.join(PROJECT_ROOT, "data_clean")
    JAPAN_PATH = os.path.join(DATA_CLEAN_DIR, "train_japan.csv")
    PHILIPPINES_PATH = os.path.join(DATA_CLEAN_DIR, "train_philippines.csv")
    PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "03_lightgbm")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Best Params from Optuna
    # JAPAN RMSLE: 0.1011
    params_japan = {
        'learning_rate': 0.01563892224383956,
        'num_leaves': 22,
        'max_depth': 5,
        'min_child_samples': 11,
        'subsample': 0.8010934630069068,
        'colsample_bytree': 0.830133521161127,
        'reg_alpha': 0.004931727017405291,
        'reg_lambda': 1.3691380094884889e-08,
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000,
        'verbosity': -1,
        'boosting_type': 'gbdt'
    }

    # PHILIPPINES RMSLE: 0.2387
    params_philippines = {
        'learning_rate': 0.021026032262560463,
        'num_leaves': 136,
        'max_depth': 13,
        'min_child_samples': 72,
        'subsample': 0.7410531102911122,
        'colsample_bytree': 0.925811310318166,
        'reg_alpha': 4.002456541286607e-07,
        'reg_lambda': 6.030902568206356e-07,
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000,
        'verbosity': -1,
        'boosting_type': 'gbdt'
    }

    datasets = [
        ('Japan', JAPAN_PATH, params_japan),
        ('Philippines', PHILIPPINES_PATH, params_philippines)
    ]
    
    results = {}

    for name, path, params in datasets:
        print(f"\n--- Training LightGBM for {name} ---")
        df = pd.read_csv(path)
        
        target_col = 'construction_cost_per_m2_usd'
        y = df[target_col]
        X = df.drop(columns=[target_col])
        y_log = np.log1p(y)
        
        # Categorical handling
        obj_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')
            
        # 5-Fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        rmses = []
        r2s = []
        
        # Manual CV to handle LGBM properly with categories
        # sklearn's cross_validate with LGBMRegressor usually works fine but manual gives control
        
        model = lgb.LGBMRegressor(**params)
        
        # Cross Validate
        cv_res = cross_validate(
            model, X, y_log, cv=kf, 
            scoring=['neg_mean_squared_error', 'r2'],
            return_estimator=True
        )
        
        rmse_scores = np.sqrt(-cv_res['test_neg_mean_squared_error'])
        mean_rmsle = rmse_scores.mean()
        std_rmsle = rmse_scores.std()
        mean_r2 = cv_res['test_r2'].mean()
        
        print(f"{name} CV RMSLE: {mean_rmsle:.4f} (+/- {std_rmsle:.4f})")
        print(f"{name} CV R2: {mean_r2:.4f}")
        
        # Train Full Model for Feature Importance & Saving
        final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X, y_log)
        
        # Save Model
        model_path = os.path.join(SCRIPT_DIR, f"model_{name}_train.joblib")
        joblib.dump(final_model, model_path)
        print(f"Saved model to {model_path}")
        
        # Plot Feature Importance
        lgb.plot_importance(final_model, max_num_features=20, importance_type='split', title=f'{name} Feature Importance (Split)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'lgbm_importance_{name}.png'))
        plt.close()
        
        # Actual vs Predicted Plot (approximate using last predict)
        preds_log = final_model.predict(X)
        
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_log, y=preds_log, alpha=0.3)
        plt.plot([y_log.min(), y_log.max()], [y_log.min(), y_log.max()], 'r--')
        plt.title(f'{name} (LGBM): Actual vs Predicted')
        plt.xlabel('Actual Log Cost')
        plt.ylabel('Predicted Log Cost')
        plt.savefig(os.path.join(PLOT_DIR, f'lgbm_performance_{name}.png'))
        plt.close()
        
        results[name] = {'rmsle': mean_rmsle, 'r2': mean_r2}

    return results

if __name__ == "__main__":
    train_lgbm()
