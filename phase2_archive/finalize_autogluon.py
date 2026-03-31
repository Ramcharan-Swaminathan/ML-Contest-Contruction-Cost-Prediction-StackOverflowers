import pandas as pd
import numpy as np
import os
from autogluon.tabular import TabularPredictor
import warnings

warnings.filterwarnings('ignore')

# --- Config ---
TEST_DATA = 'evaluation_dataset/evaluation_tabular_no_target.csv'
TEST_FEATS = 'evaluation_dataset/image_features_test.csv'
# Point to the sub-fit directory where models were actually saved
MODEL_PATH = 'autogluon_m30_tabular/ds_sub_fit/sub_fit_ho'
OUTPUT_FILE = 'submission030.csv'

def finalize_and_predict():
    print(f"Loading Predictor from {MODEL_PATH}...")
    try:
        predictor = TabularPredictor.load(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print("Checking Leaderboard...")
    lb = predictor.leaderboard(silent=True)
    print(lb[['model', 'score_val']])

    # Check if WeightedEnsemble exists
    has_ensemble = lb['model'].str.contains('WeightedEnsemble').any()
    if not has_ensemble:
        print("WeightedEnsemble not found. Manually fitting ensemble...")
        try:
            predictor.fit_weighted_ensemble()
            print("Ensemble Fit Complete.")
        except Exception as e:
            print(f"Ensembling failed: {e}. Will use best single model.")

    # Prepare Test Data
    print("Loading Test Data...")
    test_df = pd.read_csv(TEST_DATA)
    if os.path.exists(TEST_FEATS):
        feats = pd.read_csv(TEST_FEATS)
        test_df = test_df.merge(feats, on='data_id', how='left')

    # Predict
    print("Generating Predictions (using best model)...")
    # predictor.predict automatically uses the best model (ensemble if available)
    preds_log = predictor.predict(test_df)
    preds = np.expm1(preds_log)
    
    # Save
    sub = pd.DataFrame()
    if 'data_id' in test_df.columns:
        sub['data_id'] = test_df['data_id']
    else:
        sub['data_id'] = test_df.index
        
    sub['construction_cost_per_m2_usd'] = preds
    sub.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    finalize_and_predict()
