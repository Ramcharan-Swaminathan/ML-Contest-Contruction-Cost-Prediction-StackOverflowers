import pandas as pd
import numpy as np
import os
from autogluon.tabular import TabularPredictor
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
TEST_DATA = 'evaluation_dataset/evaluation_tabular_no_target.csv'
TEST_FEATS = 'evaluation_dataset/image_features_test.csv'
MODEL_PATH = 'autogluon_m30_tabular'
OUTPUT_FILE = 'submission030.csv'

def predict_autogluon():
    print("Loading Test Data...")
    test_df = pd.read_csv(TEST_DATA)
    
    if os.path.exists(TEST_FEATS):
        print(f"Merging with features from {TEST_FEATS}...")
        feats = pd.read_csv(TEST_FEATS)
        test_df = test_df.merge(feats, on='data_id', how='left')
    else:
        print("Warning: Image features not found!")

    # 2. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    try:
        predictor = TabularPredictor.load(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Predict columns (dropping ID)
    # TabularPredictor ignores extra columns automatically, but good to be safe.
    print("Generating Predictions...")
    preds_log = predictor.predict(test_df)
    preds = np.expm1(preds_log)
    
    # 4. Save
    sub = pd.DataFrame()
    if 'data_id' in test_df.columns:
        sub['data_id'] = test_df['data_id']
    else:
        print("Warning: data_id not found, using index.")
        sub['data_id'] = test_df.index
        
    sub['construction_cost_per_m2_usd'] = preds
    sub.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    predict_autogluon()
