import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train_random_forest():
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_CLEAN_DIR = os.path.join(PROJECT_ROOT, "data_clean")
    JAPAN_PATH = os.path.join(DATA_CLEAN_DIR, "train_japan.csv")
    PHILIPPINES_PATH = os.path.join(DATA_CLEAN_DIR, "train_philippines.csv")
    PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "02_random_forest")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Load Data
    print("Loading data...")
    df_jp = pd.read_csv(JAPAN_PATH)
    df_ph = pd.read_csv(PHILIPPINES_PATH)
    
    datasets = [('Japan', df_jp), ('Philippines', df_ph)]
    
    results = {}

    for name, df in datasets:
        print(f"\n--- Training Random Forest for {name} ---")
        
        target_col = 'construction_cost_per_m2_usd'
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Log-transform target
        y_log = np.log1p(y)
        
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"Num features: {len(numeric_features)} {numeric_features}")
        
        # Pipeline: OH Encode cats, pass-through nums (scaling not needed for RF)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            verbose_feature_names_out=False
        )
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        # CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_results = cross_validate(
            model, X, y_log, cv=kf,
            scoring=['neg_mean_squared_error', 'r2'],
            return_train_score=True,
            return_estimator=True
        )
        
        rmse_scores = np.sqrt(-cv_results['test_neg_mean_squared_error'])
        mean_rmsle = rmse_scores.mean()
        std_rmsle = rmse_scores.std()
        mean_r2 = cv_results['test_r2'].mean()
        
        print(f"{name} CV RMSLE: {mean_rmsle:.4f} (+/- {std_rmsle:.4f})")
        print(f"{name} CV R2: {mean_r2:.4f}")
        
        results[name] = {'rmsle': mean_rmsle, 'r2': mean_r2}
        
        # Feature Importance
        # Train on full data
        model.fit(X, y_log)
        
        # Extract importance
        regressor = model.named_steps['regressor']
        encoder_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(encoder_feature_names)
        
        importances = regressor.feature_importances_
        feature_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_imp_df = feature_imp_df.sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp_df, y='feature', x='importance', palette='viridis')
        plt.title(f'{name}: Top 20 Feature Importance (RF)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'rf_importance_{name}.png'))
        plt.close()
        
        # Actual vs Predicted
        y_pred_log = model.predict(X)
        
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_log, y=y_pred_log, alpha=0.3)
        plt.plot([y_log.min(), y_log.max()], [y_log.min(), y_log.max()], 'r--')
        plt.title(f'{name} (RF): Actual vs Predicted')
        plt.xlabel('Actual Log Cost')
        plt.ylabel('Predicted Log Cost')
        plt.savefig(os.path.join(PLOT_DIR, f'rf_performance_{name}.png'))
        plt.close()

    return results

if __name__ == "__main__":
    train_random_forest()
