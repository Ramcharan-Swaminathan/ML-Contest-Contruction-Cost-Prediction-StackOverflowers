import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression_improved():
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_CLEAN_DIR = os.path.join(PROJECT_ROOT, "data_clean")
    JAPAN_PATH = os.path.join(DATA_CLEAN_DIR, "train_japan.csv")
    PHILIPPINES_PATH = os.path.join(DATA_CLEAN_DIR, "train_philippines.csv")
    PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "01b_linear_regression_improved")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Load Data
    print("Loading data...")
    df_jp = pd.read_csv(JAPAN_PATH)
    df_ph = pd.read_csv(PHILIPPINES_PATH)
    
    datasets = [('Japan', df_jp), ('Philippines', df_ph)]
    
    results = {}

    for name, df in datasets:
        print(f"\n--- Training Improved Linear Regression for {name} ---")
        
        # Determine target and features
        target_col = 'construction_cost_per_m2_usd'
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Feature Engineering: Address Non-Linearity
        # 0. Drop Capital Features for Philippines (if present)
        if name == 'Philippines':
            drop_cols_ph = ['is_capital', 'us_cpi_x_is_capital']
            X = X.drop(columns=drop_cols_ph, errors='ignore')
            print(f"  Dropped features for Philippines: {drop_cols_ph}")

        # 1. Log-Transform Distance
        if 'straight_distance_to_capital_km' in X.columns:
            print("  Adding log_distance feature...")
            X['log_distance'] = np.log1p(X['straight_distance_to_capital_km'])
            
            # 2. Polynomial Distance
            print("  Adding distance_squared feature...")
            X['distance_squared'] = X['straight_distance_to_capital_km'] ** 2
        
        # Log-transform target
        y_log = np.log1p(y)
        
        # Identify Columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"Num features: {len(numeric_features)} {numeric_features}")
        print(f"Cat features: {len(categorical_features)}")
        
        # Preprocessing Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            verbose_feature_names_out=False
        )
        
        # Model Pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        # Cross-Validation
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
        
        # Visualisation
        model.fit(X, y_log)
        y_pred_log = model.predict(X)
        y_pred = np.expm1(y_pred_log)
        residuals = y_log - y_pred_log
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_log, y=y_pred_log, alpha=0.3)
        plt.plot([y_log.min(), y_log.max()], [y_log.min(), y_log.max()], 'r--')
        plt.title(f'{name} (Improved): Actual vs Predicted')
        plt.xlabel('Actual Log Cost')
        plt.ylabel('Predicted Log Cost')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_pred_log, y=residuals, alpha=0.3)
        plt.axhline(0, color='r', linestyle='--')
        plt.title(f'{name} (Improved): Residuals')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'lr_improved_performance_{name}.png'))
        plt.close()

    return results

if __name__ == "__main__":
    train_linear_regression_improved()
