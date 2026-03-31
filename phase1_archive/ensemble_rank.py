import pandas as pd
import numpy as np
import argparse
from scipy.stats import rankdata

def rank_ensemble(args):
    # Load Submissions
    sub09 = pd.read_csv(args.sub09)
    sub10 = pd.read_csv(args.sub10)
    
    pred09 = sub09['construction_cost_per_m2_usd'].values
    pred10 = sub10['construction_cost_per_m2_usd'].values
    
    # 1. Convert to Ranks (normalized 0 to 1)
    rank09 = rankdata(pred09) / len(pred09)
    rank10 = rankdata(pred10) / len(pred10)
    
    # 2. Weighted Average of Ranks
    # We trust LGBM ranking much more, but let DL influence the order
    w09 = 0.85
    w10 = 0.15
    avg_rank = (w09 * rank09) + (w10 * rank10)
    
    # 3. Map back to Target Distribution (Quantile Mapping)
    # Use Method 09's distribution as the ground truth for "scale"
    # Logic: "The sample with the N-th highest average rank gets the N-th highest value from Pred09"
    
    # Get the values of pred09 sorted
    sorted_pred09 = np.sort(pred09)
    
    # Get the indices that would sort the avg_rank array
    argsort_avg_rank = np.argsort(avg_rank)
    
    # Create an empty array for final predictions
    final_pred = np.zeros_like(pred09)
    
    # Assign values:
    # average_rank[i] is minimal -> it should get sorted_pred09[0]
    # We map the sorted values to the positions defined by the ensemble ranking
    
    # The item with the smallest rank (argsort[0]) gets the smallest value
    final_pred[argsort_avg_rank] = sorted_pred09
    
    # Save
    submission = pd.DataFrame()
    if 'data_id' in sub09.columns:
        submission['data_id'] = sub09['data_id']
    submission['construction_cost_per_m2_usd'] = final_pred
    
    submission.to_csv(args.output, index=False)
    print(f"Rank Ensemble saved to {args.output}")
    print(f"Blending Ranks: {w09} * M09 + {w10} * M10")
    print("Mapped to distribution of Method 09")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub09', type=str, default='submission009.csv')
    parser.add_argument('--sub10', type=str, default='submission010.csv')
    parser.add_argument('--output', type=str, default='submission_method12.csv')
    
    args = parser.parse_args()
    rank_ensemble(args)
