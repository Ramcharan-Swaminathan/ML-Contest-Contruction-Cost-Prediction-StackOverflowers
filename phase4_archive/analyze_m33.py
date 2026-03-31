import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, skew, kurtosis
from scipy.cluster import hierarchy
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Config ---
TRAIN_TAB = 'dataset/train_tabular.csv'
TRAIN_FEATS = 'dataset/image_features_train.csv'
REPORT_FILE = 'feature_selection_report.md'
JSON_FILE = 'golden_features.json'

def analyze_target(df):
    target = df['construction_cost_per_m2_usd']
    log_target = np.log1p(target)
    
    report = "## 1. Target Variable Analysis\n"
    report += f"- **Count**: {len(target)}\n"
    report += f"- **Skew**: {skew(target):.4f} (Raw) vs {skew(log_target):.4f} (Log)\n"
    report += f"- **Kurtosis**: {kurtosis(target):.4f} (Raw) vs {kurtosis(log_target):.4f} (Log)\n"
    report += "- **Conclusion**: " 
    if abs(skew(log_target)) < abs(skew(target)):
        report += "Log-transform significantly improves normality. We will use `log1p`.\n\n"
    else:
        report += "Log-transform not effective (Unlikely). Check data.\n\n"
    return report

def analyze_image_features(df):
    # Filter image cols
    img_cols = [c for c in df.columns if (c.startswith('s2_') or c.startswith('viirs_')) and not c.endswith('_name')]
    
    report = "## 2. Image Feature Analysis\n"
    report += f"- **Total Image Features**: {len(img_cols)}\n"
    
    if len(img_cols) == 0:
        report += "- **Error**: No image features found.\n"
        return report, []

    data = df[img_cols].fillna(0)
    
    # 1. Hierarchical Clustering for Redundancy
    corr = data.corr(method='spearman')
    corr_linkage = hierarchy.ward(corr)
    
    # Identify clusters
    cluster_ids = hierarchy.fcluster(corr_linkage, t=1.0, criterion='distance')
    cluster_id_to_feature_ids = {}
    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id not in cluster_id_to_feature_ids:
            cluster_id_to_feature_ids[cluster_id] = []
        cluster_id_to_feature_ids[cluster_id].append(img_cols[i])

    report += "- **Redundancy Check**:\n"
    selected_representatives = []
    
    for cluster_id, features in cluster_id_to_feature_ids.items():
        if len(features) > 1:
            # Pick the one with highest correlation to target
            corrs = [abs(spearmanr(df[f], df['log_cost'])[0]) for f in features]
            best_feat = features[np.argmax(corrs)]
            report += f"    - **Cluster {cluster_id}**: {len(features)} features ({', '.join(features[:3])}...). Keeping **{best_feat}**.\n"
            selected_representatives.append(best_feat)
        else:
            selected_representatives.append(features[0])
            
    report += f"\n- **Reduced Set**: {len(selected_representatives)} features (from {len(img_cols)}).\n\n"
    return report, selected_representatives

def analyze_importance(df, candidate_features):
    report = "## 3. Feature Importance (Random Forest)\n"
    
    # Prep Data
    X = df[candidate_features].fillna(0)
    # One-Hot encode if valid object/cat types exist and are in candidate_features
    # Assuming candidate_features are mostly numeric for now, but let's handle obj
    X = pd.get_dummies(X, dummy_na=True)
    y = df['log_cost']
    
    # Train RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Permutation Importance
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Organizing
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    report += "| Feature | Importance | Std |\n|---|---|---|\n"
    golden = []
    
    cumulative = 0
    for _, row in importances.iterrows():
        report += f"| {row['feature']} | {row['importance']:.4f} | {row['std']:.4f} |\n"
        if row['importance'] > 0.001: # Threshold
            golden.append(row['feature'])
            
    report += f"\n- **Selected Features**: {len(golden)} (Importance > 0.001)\n"
    return report, golden

def run_analysis():
    # Load
    print("Loading Data...")
    df = pd.read_csv(TRAIN_TAB)
    if os.path.exists(TRAIN_FEATS):
        feats = pd.read_csv(TRAIN_FEATS)
        df = df.merge(feats, on='data_id', how='left')
    
    df['log_cost'] = np.log1p(df['construction_cost_per_m2_usd'])
    
    full_report = "# Method 33: Deep EDA & Feature Selection Report\n\n"
    
    # 1. Target
    full_report += analyze_target(df)
    
    # 2. Image Features
    img_report, img_golden = analyze_image_features(df)
    full_report += img_report
    
    # 3. Tabular Features Selection
    # Hand-picked basics + Image Golden
    # Note: excluding 'quarter_label' logic for now as it was weak
    tab_cols = ['deflated_gdp_usd', 'us_cpi', 'straight_distance_to_capital_km', 
                'access_to_port', 'access_to_airport', 'access_to_highway', 'access_to_railway',
                'seismic_hazard_zone', 'flood_risk_class', 'tropical_cyclone_wind_risk', 
                'tornadoes_wind_risk', 'koppen_climate_zone']
    # One hot encoding happens inside analyze_importance, so we pass raw cols
    
    # Combined Candidates
    candidates = tab_cols + img_golden
    
    # 4. Importance
    imp_report, final_golden = analyze_importance(df, candidates)
    full_report += imp_report
    
    # Save
    with open(REPORT_FILE, 'w') as f:
        f.write(full_report)
    
    with open(JSON_FILE, 'w') as f:
        json.dump(final_golden, f, indent=4)
        
    print(f"Analysis Complete. Report: {REPORT_FILE}, Golden List: {JSON_FILE}")

if __name__ == "__main__":
    run_analysis()
