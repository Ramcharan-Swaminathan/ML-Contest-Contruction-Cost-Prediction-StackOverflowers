import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# --- Config ---
TRAIN_OOFS = 'oofs_train.csv'
TEST_OOFS = 'oofs_test.csv'
SUB_FILE = 'submission042.csv'
SEED = 42

def run_stacking():
    print("Loading OOFs...")
    df_train = pd.read_csv(TRAIN_OOFS)
    df_test = pd.read_csv(TEST_OOFS)
    
    # Features for stacking
    features = ['m21', 'm35', 'm40']
    
    X = df_train[features]
    y = df_train['target'] # This is log1p target
    X_test = df_test[features]
    
    print("\nCorrelation Matrix of Base Models:")
    print(X.corr())
    
    # Meta-Model
    # Ridge is standard. Positive constraint is good for ensembling.
    # scikit-learn Ridge doesn't easily support positive=True until recent versions.
    # LinearRegression does with positive=True? No.
    # Lasso does.
    # Let's use simple Ridge, if weights are negative, we clamp or use NNLS.
    # Actually, simpler: LinearRegression? No, Ridge for stability.
    
    model = Ridge(alpha=0.1, random_state=SEED)
    
    # CV for Stacking Score
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], preds))
        scores.append(rmse)
        
    print(f"\nStacking CV RMSLE: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    
    # Final Fit
    model.fit(X, y)
    print("\nLearned Weights (Coefficients):")
    for f, w in zip(features, model.coef_):
        print(f"  {f}: {w:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    
    # Predict
    final_preds_log = model.predict(X_test)
    final_preds = np.expm1(final_preds_log)
    
    sub = pd.DataFrame({'data_id': df_test['data_id'], 'construction_cost_per_m2_usd': final_preds})
    sub.to_csv(SUB_FILE, index=False)
    print(f"Saved {SUB_FILE}")

if __name__ == "__main__":
    run_stacking()
