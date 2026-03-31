import pandas as pd
import numpy as np
import argparse
import os

def ensemble(args):
    """
    Blends predictions from multiple submission files using weighted averaging in log-space.
    """
    files = args.files
    weights = args.weights
    output_path = args.output_path
    
    if len(files) != len(weights):
        raise ValueError("Number of files and weights must match")
        
    print(f"Ensembling files: {files}")
    print(f"Weights: {weights}")
    
    normalization = sum(weights)
    weights = [w / normalization for w in weights]
    print(f"Normalized Weights: {weights}")
    
    dfs = []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}")
        dfs.append(pd.read_csv(f))
        
    # Check if data_ids align
    base_id = dfs[0]['data_id']
    for i, df in enumerate(dfs[1:]):
        if not df['data_id'].equals(base_id):
            raise ValueError(f"file {files[i+1]} data_id column does not match {files[0]}")
            
    # Compute Weighted Average in Log Space
    # Prediction values in CSV are actual values (expm1'd), so we need to log1p them back
    # Average them, then expm1 back.
    # Why? Because metric is RMSLE. Averaging in log-space minimizes RMSLE better than linear space.
    
    log_sum = np.zeros(len(base_id))
    
    for i, df in enumerate(dfs):
        preds = df['construction_cost_per_m2_usd'].values
        # Handle potential negatives (though shouldn't exist)
        preds = np.maximum(preds, 0)
        log_preds = np.log1p(preds)
        log_sum += weights[i] * log_preds
        
    final_preds = np.expm1(log_sum)
    
    submission = pd.DataFrame()
    submission['data_id'] = base_id
    submission['construction_cost_per_m2_usd'] = final_preds
    
    submission.to_csv(output_path, index=False)
    print(f"Ensemble saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults to blending 004 and 005
    parser.add_argument('--files', nargs='+', default=['submission005.csv', 'submission004.csv'], help='List of submission CSV files')
    parser.add_argument('--weights', nargs='+', type=float, default=[0.6, 0.4], help='List of weights for each file')
    parser.add_argument('--output_path', type=str, default='submission_ensemble.csv')
    
    args = parser.parse_args()
    ensemble(args)
