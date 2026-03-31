# ==========================================
# Solafune Construction Cost Prediction
# Workflow: Single CSV -> Train/Test Split -> LightGBM (GPU)
# ==========================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ==========================================
# 1. DATA LOADING & SPLITTING
# ==========================================
# Replace 'your_data.csv' with the actual filename you have
FILE_PATH = 'dataset/train_tabular.csv' 

print("Loading dataset...")
df_full = pd.read_csv(FILE_PATH)

# Basic Sanity Check: Ensure target variable exists and isn't null
df_full = df_full.dropna(subset=['construction_cost_per_m2_usd'])

print(f"Total Samples: {len(df_full)}")

# SPLIT STRATEGY:
# We use an 80/20 split. 
# random_state=42 ensures the split is reproducible (same rows every time).
train_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42)

print(f"Training Samples: {len(train_df)}")
print(f"Hold-out Test Samples: {len(test_df)}")

# ==========================================
# 2. EDA (Exploratory Data Analysis)
# ==========================================
# IMPORTANT: Only look at training data statistics to avoid "Data Leakage"
# The target variable is 'construction_cost_per_m2_usd' 

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_df['construction_cost_per_m2_usd'], kde=True)
plt.title("Original Target (Train Only)")

# Log-transform because the metric is RMSLE 
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(train_df['construction_cost_per_m2_usd']), kde=True)
plt.title("Log-Transformed Target (Train Only)")
plt.show()

# ==========================================
# 3. PREPROCESSING
# ==========================================
# ==========================================
# CORRECTED PREPROCESSING FUNCTION
# ==========================================

def preprocess_data(df):
    # Drop unique identifiers and image file paths
    # We keep 'country' and geographic data
    drop_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name']
    
    # Separate target (y) from features (X)
    if 'construction_cost_per_m2_usd' in df.columns:
        y = np.log1p(df['construction_cost_per_m2_usd']) # Log transform for RMSLE
        X = df.drop(columns=drop_cols + ['construction_cost_per_m2_usd'])
    else:
        y = None
        X = df.drop(columns=drop_cols)
    
    # 1. DEFINE CATEGORICAL COLUMNS
    # We added the 'access_to_*' columns here to fix the ValueError
    cat_cols = [
        'geolocation_name', 'quarter_label', 'country', 'region_economic_classification',
        'seismic_hazard_zone', 'flood_risk_class', 'tropical_cyclone_wind_risk', 
        'tornadoes_wind_risk', 'koppen_climate_zone', 'developed_country', 'landlocked',
        'access_to_airport', 'access_to_port', 'access_to_highway', 'access_to_railway'
    ]
    
    # 2. CAST TO CATEGORY
    for col in cat_cols:
        if col in X.columns:
            # Force conversion to category dtype so LightGBM handles it natively
            X[col] = X[col].astype('category')

    # 3. SAFETY CHECK (Catch-all for any other object columns)
    # This ensures no "object" columns slip through 
    for col in X.select_dtypes(include='object').columns:
        print(f"Warning: Found remaining object column '{col}'. Casting to category.")
        X[col] = X[col].astype('category')
            
    return X, y

# Prepare Training Data
X_train, y_train = preprocess_data(train_df)

# Prepare Test Data (The hold-out set)
X_test, y_test = preprocess_data(test_df)

print("\nFeatures selected for training:")
print(X_train.columns.tolist())

# ==========================================
# 4. LIGHTGBM MODELING (GPU ENABLED)
# ==========================================

# Create LGBM Datasets
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

# Hyperparameters
params = {
    'objective': 'regression',
    'metric': 'rmse', # Optimizing RMSE on Log data = Optimizing RMSLE on original
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'device': 'cuda', # CUDA ENABLED [cite: 78]
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbose': -1,
    'seed': 42
}

print(f"\nStarting Training on {len(X_train)} samples with GPU...")

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=10000,
    valid_sets=[lgb_train, lgb_eval],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
)

# ==========================================
# 5. EVALUATION & ANALYSIS
# ==========================================

# Predict on the hold-out test set
# The model outputs log(cost), so these are log predictions
log_preds = model.predict(X_test, num_iteration=model.best_iteration)

# Calculate RMSLE
# Since both y_test and log_preds are already in log scale, RMSE here IS the RMSLE
rmsle_score = np.sqrt(mean_squared_error(y_test, log_preds))

print(f"\n========================================")
print(f"FINAL RESULT ON HOLD-OUT SET")
print(f"RMSLE Score: {rmsle_score:.5f}")
print(f"========================================")

# Visualize Predictions vs Actuals
plt.figure(figsize=(8, 8))
plt.scatter(y_test, log_preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log Cost')
plt.ylabel('Predicted Log Cost')
plt.title(f'Prediction Accuracy (RMSLE: {rmsle_score:.4f})')
plt.show()

# Feature Importance
lgb.plot_importance(model, max_num_features=15, importance_type='gain', figsize=(10, 6))
plt.title("Top 15 Most Important Features")
plt.show()

# Example: Inspecting a few actual dollar predictions
# We must use expm1 to reverse the log transformation [cite: 65]
actual_usd = np.expm1(y_test.values[:5])
pred_usd = np.expm1(log_preds[:5])

print("\nSample Predictions (USD/m2):")
for a, p in zip(actual_usd, pred_usd):
    print(f"Actual: ${a:,.2f} | Predicted: ${p:,.2f} | Diff: ${p-a:,.2f}")