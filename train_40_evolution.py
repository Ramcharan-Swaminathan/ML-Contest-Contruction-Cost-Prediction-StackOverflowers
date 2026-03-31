import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
import random
import copy
from sklearn.model_selection import KFold

# --- Config ---
TRAIN_TAB = 'dataset/train_tabular.csv'
TEST_TAB = 'evaluation_dataset/evaluation_tabular_no_target.csv'
TRAIN_FEATS = 'dataset/image_features_train.csv'
TEST_FEATS = 'evaluation_dataset/image_features_test.csv'
PSEUDO_LABEL_FILE = 'phase1_archive/sub_ens_kestav.csv' # Teacher Labels
SUBMISSION_FILE = 'submission040.csv'
FEATURE_FILE = 'evolved_features.json'
SEED = 42

# GA Config
POP_SIZE = 20
GENERATIONS = 15
MUTATION_RATE = 0.1
ELITISM = 2

def load_data():
    print("Loading Data...")
    df_train = pd.read_csv(TRAIN_TAB)
    df_test = pd.read_csv(TEST_TAB)
    
    if os.path.exists(TRAIN_FEATS):
        f_train = pd.read_csv(TRAIN_FEATS)
        f_test = pd.read_csv(TEST_FEATS)
        df_train = df_train.merge(f_train, on='data_id', how='left')
        df_test = df_test.merge(f_test, on='data_id', how='left')
        
    # Pseudo Labeling (Use Expanded Data for Evolution)
    df_pseudo = pd.read_csv(PSEUDO_LABEL_FILE)
    df_test_labeled = df_test.copy()
    df_test_labeled = df_test_labeled.merge(df_pseudo, on='data_id', how='left')
    
    df_combined = pd.concat([df_train, df_test_labeled], axis=0).reset_index(drop=True)
    
    y = np.log1p(df_combined['construction_cost_per_m2_usd'])
    
    # Candidate Features
    # Exclude ID, Target, Geolocation (Scalar model handles geo via features, usually)
    # Actually, keep strictly numeric candidate features for GA stability
    candidates = [c for c in df_combined.columns if c not in ['data_id', 'construction_cost_per_m2_usd', 'geolocation_name']]
    # Filter for numeric
    candidates = [c for c in candidates if np.issubdtype(df_combined[c].dtype, np.number)]
    
    X = df_combined[candidates].copy()
    X_test = df_test[candidates].copy()
    
    print(f"Candidate Features ({len(candidates)}): {candidates[:5]} ...")
    
    return X, y, candidates, X_test

def evaluate(features, X, y):
    if len(features) == 0: return 999.0
    
    X_subset = X[features]
    
    # Fast LGBM 
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 300, # Fast check
        'learning_rate': 0.05,
        'num_leaves': 16,
        'max_depth': 6,
        'seed': SEED
    }
    
    scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for train_idx, val_idx in kf.split(X_subset, y):
        model = lgb.LGBMRegressor(**params)
        model.fit(X_subset.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X_subset.iloc[val_idx])
        rmse = np.sqrt(np.mean((preds - y.iloc[val_idx])**2))
        scores.append(rmse)
        
    return np.mean(scores)

# --- GA Logic ---
def init_population(features, n):
    pop = []
    for _ in range(n):
        # 10% chance to include a feature initially
        mask = [random.random() < 0.1 for _ in features]
        # Ensure at least 3 features
        if sum(mask) < 3:
            for i in range(3): mask[i] = True 
        pop.append(mask)
    return pop

def decode_mask(mask, features):
    return [f for i, f in enumerate(features) if mask[i]]

def run_ga():
    X, y, candidates, X_test = load_data()
    
    population = init_population(candidates, POP_SIZE)
    # Ensure our "Domain Expert" set is in the initial population as a baseline!
    domain_set = ["deflated_gdp_usd", "viirs_mean", "s2_B11_mean", "s2_B8A_mean", "s2_B1_mean", "straight_distance_to_capital_km"]
    domain_mask = [f in domain_set for f in candidates]
    population[0] = domain_mask
    
    best_score = 999.0
    best_mask = []
    
    print("Starting Evolution...")
    
    for gen in range(GENERATIONS):
        scores = []
        for i, mask in enumerate(population):
            feats = decode_mask(mask, candidates)
            score = evaluate(feats, X, y)
            scores.append((score, mask))
            
        scores.sort(key=lambda x: x[0]) # Minimize RMSE
        
        current_best = scores[0][0]
        if current_best < best_score:
            best_score = current_best
            best_mask = scores[0][1]
            print(f"Generation {gen}: New Best RMSE {best_score:.5f} with {len(decode_mask(best_mask, candidates))} features")
        else:
            print(f"Generation {gen}: Best RMSE {current_best:.5f}")
            
        # Selection & Elitism
        next_pop = [s[1] for s in scores[:ELITISM]]
        
        # Crossover & Mutation
        while len(next_pop) < POP_SIZE:
            # Tournament Selection
            p1 = min(random.sample(scores, 3), key=lambda x: x[0])[1]
            p2 = min(random.sample(scores, 3), key=lambda x: x[0])[1]
            
            # Uniform Crossover
            child = []
            for i in range(len(candidates)):
                if random.random() < 0.5: child.append(p1[i])
                else: child.append(p2[i])
                
            # Mutation
            for i in range(len(candidates)):
                if random.random() < MUTATION_RATE:
                    child[i] = not child[i]
                    
            if sum(child) > 0:
                next_pop.append(child)
                
        population = next_pop
        
    final_features = decode_mask(best_mask, candidates)
    print(f"Evolution Complete. Best Features ({len(final_features)}): {final_features}")
    
    # Save Features
    with open(FEATURE_FILE, 'w') as f:
        json.dump(final_features, f)
        
    # Train Final Model
    print("Training Final M40 Model...")
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.02, # Slower for final
        'num_leaves': 31,
        'max_depth': 8,
        'seed': SEED
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X[final_features], y)
    preds = model.predict(X_test[final_features])
    
    sub = pd.DataFrame({'data_id': pd.read_csv(TEST_TAB)['data_id'], 'construction_cost_per_m2_usd': np.expm1(preds)})
    sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved {SUBMISSION_FILE}")

if __name__ == "__main__":
    run_ga()
