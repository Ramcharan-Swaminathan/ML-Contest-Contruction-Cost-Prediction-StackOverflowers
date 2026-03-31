# Construction Cost Prediction — Complete Project Report

> **Project**: Predicting `construction_cost_per_m2_usd` for construction sites across Japan and the Philippines.
> **Metric**: RMSLE (Root Mean Squared Logarithmic Error)
> **Dataset**: 1,024 training samples (multi-modal: tabular + satellite imagery)
> **Best Score Achieved**: **0.21899** (RMSLE on evaluation leaderboard)
> **Date**: 2025-12 to 2026-03

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Description](#2-dataset-description)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Data Cleaning & Feature Engineering](#4-data-cleaning--feature-engineering)
5. [Phase 1: Baseline & Core Model Development (Methods 1–29)](#5-phase-1-baseline--core-model-development-methods-129)
6. [Phase 2: AutoML & Graph Neural Networks (Methods 30–31)](#6-phase-2-automl--graph-neural-networks-methods-3031)
7. [Phase 3: Supervised Contrastive Learning (Method 32)](#7-phase-3-supervised-contrastive-learning-method-32)
8. [Phase 4: Deep Feature Selection & Domain Models (Methods 33–36)](#8-phase-4-deep-feature-selection--domain-models-methods-3336)
9. [Phase 5: Blending, Stacking & Final Optimization (Methods 37–48)](#9-phase-5-blending-stacking--final-optimization-methods-3748)
10. [Results Summary & Leaderboard Scores](#10-results-summary--leaderboard-scores)
11. [Feature Lists Used](#11-feature-lists-used)
12. [Key Findings & Insights](#12-key-findings--insights)
13. [Future Plans & Potential Improvements](#13-future-plans--potential-improvements)
14. [Project Structure](#14-project-structure)

---

## 1. Problem Statement

The goal is to predict `construction_cost_per_m2_usd` (USD per square meter construction cost) for construction sites located in **Japan** and the **Philippines** using a combination of:

- **Tabular data**: Economic, geographic, climatic, and infrastructure features.
- **Satellite imagery**: Sentinel-2 (12-band multispectral) and VIIRS (nighttime radiance) composite images.

The evaluation metric is **RMSLE** (Root Mean Squared Logarithmic Error), which penalizes under-predictions more heavily and is scale-invariant — appropriate for a target variable spanning a wide range ($45–$3,628 USD/m²).

### Key Challenge

The dataset exhibits a **severe bimodal distribution** — Japan has costs ~$2,000/m² while the Philippines averages ~$160/m². This is essentially a **switching regression** problem where `country` is the hard separator. A single global model risks underfitting both distributions.

---

## 2. Dataset Description

### 2.1 Data Profile

| Attribute | Details |
|---|---|
| **Total Observations** | 1,024 (training) |
| **Target Variable** | `construction_cost_per_m2_usd` — Continuous, range $45–$3,628 |
| **Countries** | Japan (568 samples), Philippines (456 samples) |
| **Temporal Scope** | 2019–2024 (pre-to-post pandemic economic shifts) |
| **Missingness** | ~0.04% — Only `tropical_cyclone_wind_risk` has 4 missing values |
| **Test Set** | ~1,024 evaluation samples (no target provided) |

### 2.2 Tabular Features

| Feature Category | Features |
|---|---|
| **Economic** | `deflated_gdp_usd`, `us_cpi` |
| **Geographic** | `country`, `geolocation_name`, `region_economic_classification`, `straight_distance_to_capital_km` |
| **Temporal** | `year`, `quarter_label` |
| **Infrastructure** | `access_to_airport`, `access_to_port`, `access_to_highway`, `access_to_railway` |
| **Climate/Risk** | `seismic_hazard_zone`, `flood_risk_class`, `tropical_cyclone_wind_risk`, `tornadoes_wind_risk`, `koppen_climate_zone` |
| **Identifiers** | `data_id`, `sentinel2_tiff_file_name`, `viirs_tiff_file_name` |

### 2.3 Image Data

| Source | Type | Details |
|---|---|---|
| **Sentinel-2** | 12-band multispectral GeoTIFF | Bands B1–B12 (Blue, Green, Red, NIR, SWIR, etc.) — median composite per site |
| **VIIRS** | Single-band nighttime radiance GeoTIFF | Avg_rad band — proxy for urbanization/economic activity |

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Global Target Distribution — The "Great Divergence"

The raw target variable exhibits a **severe bimodal distribution** (Kurtosis < -1.5):

- **Philippines Manifold**: μ ≈ $160, σ ≈ $50 (lower variance)
- **Japan Manifold**: μ ≈ $2,000, σ ≈ $400 (higher variance)

There is **zero distributional overlap** between the two nations. The **Coefficient of Variation (CV)** is significantly higher in Japan, implying that unobserved variables (likely labor shortages or specific seismic regulations) drive costs more variably in the developed nation.

**Log Transformation**: Applied `y' = log(1+y)`, which reduced skewness from 0.20 to -0.05, stabilizing the variance for gradient-based optimization.

### 3.2 Elasticity of Cost to GDP

- **Japan (High GDP)**: The curve flattens — **diminishing returns**. Incremental increases in regional GDP do not yield proportional increases in construction cost.
- **Philippines (Lower GDP)**: The relationship is steeper — **supply-constrained market** where economic growth immediately drives up material prices.
- `log_gdp` is a powerful predictor but interacts non-linearly with `country`.

### 3.3 The Inflationary Mirage: Nominal vs. Real

- **Nominal** costs show a clear upward trend in costs over the 5-year period (2019–2024).
- **Real** (adjusted for US CPI): When adjusted for US CPI, the trend **flatlines**.
- **Conclusion**: The apparent rise in construction costs is largely a **monetary phenomenon** driven by global inflation (`us_cpi`), rather than structural inefficiencies in the construction sector.
- **Decision**: Include `us_cpi` as a feature.

### 3.4 Feature Orthogonality (Correlation Analysis)

- `us_cpi` is positively correlated with cost (Validation of inflation theory).
- `straight_distance_to_capital_km` shows weak negative correlation globally, but locally it has different behavior per country.

### 3.5 Logistics Anomalies: The "Remote Island" Effect

- Most data is clustered within 500km of the capital. Distinct outliers appear >1000km.
- These correspond to remote archipelagic regions (e.g., Okinawa, Mindanao extremities).
- **Decision**: Do NOT remove these outliers — they represent a valid "Logistics Premium" for sea-freight transport.
- **Feature Engineering**: Created boolean `is_remote_island = distance > 800km`.

### 3.6 Verification Results (Post-Split)

After splitting into Japan/Philippines sub-datasets:

**Target Encoding Correlation**:
| Country | Correlation (Encoded Geolocation vs. Target) |
|---|---|
| Japan | 0.6381 |
| Philippines | 0.5755 |

**Key Correlation Rankings — Japan**:
| Feature | Correlation |
|---|---|
| geolocation_name (encoded) | 0.6381 |
| is_capital | 0.4993 |
| us_cpi_x_is_capital | 0.4906 |
| straight_distance_to_capital_km | 0.0237 |
| us_cpi | -0.2457 |

**Key Correlation Rankings — Philippines**:
| Feature | Correlation |
|---|---|
| geolocation_name (encoded) | 0.5755 |
| us_cpi | 0.1200 |
| us_cpi_x_is_capital | 0.1093 |
| is_capital | 0.1090 |
| straight_distance_to_capital_km | -0.1586 |

**Outlier Scan (3σ Rule)**:
| Country | Mean | Std | 3σ Limit | Outliers | % |
|---|---|---|---|---|---|
| Japan | $1,830.27 | $309.11 | $2,757.59 | 13 | 2.29% |
| Philippines | $214.82 | $71.98 | $430.76 | 12 | 2.63% |

### 3.7 Philippines Capital Redundancy Analysis

The `is_capital` feature for Philippines was found to be **highly redundant** with `straight_distance_to_capital_km`:
- All capital observations have `distance_to_capital = 0`.
- Correlation with target: `is_capital = 0.1090` vs `distance_to_capital = -0.1586`.
- **Decision**: Drop `is_capital` and `us_cpi_x_is_capital` for Philippines only; `straight_distance_to_capital_km` already captures this information more granularly.

---

## 4. Data Cleaning & Feature Engineering

### 4.1 Data Cleaning Pipeline (`data_clean/clean_data.py`)

The data cleaning script implements the following ordered pipeline:

1. **Split by Country**: Separate Japan and Philippines datasets.
2. **Interaction Feature Creation** (done BEFORE encoding):
   - Identify capital cities: Tokyo (Japan), National Capital Region (Philippines).
   - Create `is_capital` boolean column.
   - Create `us_cpi_x_is_capital` interaction feature.
3. **Imputation**: Fill missing `tropical_cyclone_wind_risk` values with the **mode** per country.
4. **Target Encoding**: Apply `SmoothedTargetEncoder` to `geolocation_name`.
   - Formula: `smoothed = (count × mean + smoothing × global_mean) / (count + smoothing)`, with `smoothing = 10`.
   - Replaces categorical geolocation with smoothed numeric encoding.
5. **Drop Noise Columns**: Remove `country`, `data_id`, `sentinel2_tiff_file_name`, `viirs_tiff_file_name`, `deflated_gdp_usd`, `year`, `access_to_highway`.
6. **Country-Specific Feature Removal**:
   - **Philippines**: Drop `is_capital` and `us_cpi_x_is_capital` (redundant with distance_to_capital).

### 4.2 Output Datasets

| File | Country | Columns |
|---|---|---|
| `data_clean/train_japan.csv` | Japan | All cleaned features + `is_capital` + `us_cpi_x_is_capital` |
| `data_clean/train_philippines.csv` | Philippines | All cleaned features (without `is_capital` variants) |

### 4.3 Image Feature Extraction (`extract_features.py`)

A dedicated feature extraction script processes satellite imagery:

- **Sentinel-2**: 12 bands × 5 statistics (mean, std, median, min, max) = **60 features** per sample.
- **VIIRS**: 1 band × 5 statistics = **5 features** per sample.
- **Total**: 65 hand-crafted image features.
- **Processing**: Uses `rasterio` for GeoTIFF reading, parallel processing with `ProcessPoolExecutor` (8 workers).
- **Output**: `dataset/image_features_train.csv` and `evaluation_dataset/image_features_test.csv`.

### 4.4 Deep Feature Extraction (ResNet-18 CNN Embeddings)

Multiple model iterations used pre-trained ResNet-18 as a feature extractor:

1. Load pre-trained `ResNet18` (ImageNet weights).
2. Remove the final classification head.
3. Extract **512-dimensional** embedding vectors from RGB images.
   - RGB constructed from Sentinel-2 bands: B4(Red), B3(Green), B2(Blue).
   - Normalization: Percentile-based (98th percentile clip → 0–255 scaling).
4. **TTA (Test-Time Augmentation)**: 4× rotations (0°, 90°, 180°, 270°) averaged for robust embeddings.
5. **PCA Reduction**: 512 → 16 dimensions for LightGBM compatibility. Saved as `pca_model_XX.pkl`.
6. **Batch generation**: `gen_embeddings.py` generates `.npy` files for reuse.

---

## 5. Phase 1: Baseline & Core Model Development (Methods 1–29)

### Method 01: Baseline LightGBM (Global Model)

| Parameter | Value |
|---|---|
| **Architecture** | LightGBM Regressor |
| **Features** | Raw tabular features + quarter extraction |
| **Target** | log1p(construction_cost_per_m2_usd) |
| **Split** | 80/20 random train/validation |
| **Hyperparameters** | learning_rate=0.03, num_leaves=31, min_child_samples=30, reg_alpha=0.1, reg_lambda=0.1 |
| **Rounds** | 1,000 (early stopping at 50) |
| **Device** | GPU (CUDA) |
| **Submission** | `submission.csv` |

**Key Design Decisions**:
- Categorical features handled natively by LightGBM (converted to `category` dtype).
- `quarter_label` (format `YYYY-QX`) parsed to numeric `quarter` (1–4).
- Drop identifiers: `data_id`, `sentinel2_tiff_file_name`, `viirs_tiff_file_name`.

---

### Method 03: LightGBM + Hand-Crafted Image Statistics

- Added **Sentinel-2 band statistics** (mean, std, max per band) and **VIIRS statistics** (mean, max, std) as tabular features.
- Used `rasterio` to extract per-band stats from composite GeoTIFFs.
- **New features**: 36 S2 band features (12 bands × 3 stats) + 3 VIIRS features = **39 new features**.
- **Parameters**: Same as M01 but `colsample_bytree=0.7` (lower column sampling for high dimensionality).
- **Rounds**: 1,500 with early stopping at 100.
- **Submission**: `submission003.csv`

---

### Method 04: LightGBM with Refined Parameters

- Iterative parameter tuning on top of M03.
- **Submission**: `submission004.csv`

---

### Method 05: LightGBM + CNN Embeddings (ResNet-18 PCA)

- Replaced hand-crafted image stats with **ResNet-18 embeddings** (pre-trained on ImageNet).
- **Pipeline**: Sentinel-2 B4/B3/B2 → RGB Image → ResNet-18 → 512-dim embedding → PCA → 16-dim.
- PCA model saved as `pca_model_05.pkl`.
- **Submission**: `submission005.csv`

**Key Insight**: The deep embeddings capture higher-level visual patterns (urbanization density, building density, roof materials) that hand-crafted stats might miss.

---

### Method 07: 5-Fold Cross-Validation

- Applied proper **5-Fold K-Fold CV** instead of single random split.
- Used same ResNet-18 + PCA pipeline as M05.
- PCA fitted globally (on all data before split) for stability.
- Models saved per fold: `model_cv_fold_{1-5}.txt`.
- **Rounds**: 2,000 with early stopping at 100.
- **Submission**: `submission007.csv`

---

### Method 08: Test-Time Augmentation (TTA) for Embeddings

- Enhanced the ResNet-18 embedding extraction with **4× rotation TTA** (0°, 90°, 180°, 270°).
- Final embedding = mean of 4 rotated embeddings.
- **Rationale**: Satellite imagery is rotation-invariant — TTA produces more robust visual signatures.
- **Submission**: `submission008.csv`

---

### Method 09: Optuna Hyperparameter Tuning (25 Trials)

- Automated hyperparameter search using **Optuna** with **25 trials**.
- Each trial performs **5-Fold CV internally** for robust evaluation.
- Uses TTA-enhanced embeddings from M08.
- Includes `LightGBMPruningCallback` for efficient trial pruning.

**Search Space**:
| Parameter | Range |
|---|---|
| learning_rate | 0.005–0.1 |
| num_leaves | 20–100 |
| feature_fraction | 0.4–1.0 |
| bagging_fraction | 0.4–1.0 |
| bagging_freq | 1–7 |
| min_child_samples | 5–100 |
| reg_alpha | 1e-8–10.0 (log) |
| reg_lambda | 1e-8–10.0 (log) |
| colsample_bytree | 0.4–1.0 |

**Best Hyperparameters Found**:
```
learning_rate: 0.016
num_leaves: 66
feature_fraction: 0.93
bagging_fraction: 0.90
bagging_freq: 5
min_child_samples: 20
reg_alpha: 0.1
reg_lambda: 0.1
colsample_bytree: 0.8
```

- Final model trained with **5,000 rounds** and early stopping at 100.
- **Submission**: `submission009.csv`

---

### Method 10: End-to-End Deep Learning (ResNet-18 Fine-Tuning)

- **Architecture**: ResNet-18 with classification head replaced by regression head:
  ```
  ResNet-18 → 512-dim → Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 1)
  ```
- All ResNet layers unfrozen (full fine-tuning).
- **Loss**: MSE on log-transformed targets.
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-2).
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3).
- **Augmentation**: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15°), ColorJitter.
- **Epochs**: 150, batch_size=32.
- **Device**: CUDA.
- **Submission**: `submission010.csv`

**Outcome**: Deep learning alone performed poorly relative to LightGBM + embeddings due to small dataset size (1,024 samples).

---

### Method 13: Pseudo-Labeling (Self-Training)

- Used best previous submission (`submission_final.csv`) as **pseudo-labels** for the test set.
- Combined train (1,024 real samples) + test (~1,024 pseudo-labeled samples) = **~2,048 total training samples**.
- PCA fitted on combined embeddings for consistency.
- Used M09 best hyperparameters.
- Trained on combined data with **1,000 rounds** (no validation — self-training paradigm).
- **Submission**: `submission013.csv`

---

### Method 14: Deep Pseudo-Labeling

- Extended pseudo-labeling concept with deeper training pipeline.
- **Submission**: Used improved pseudo-label training.

---

### Method 15: CatBoost Regressor

- **Architecture**: CatBoost with native categorical handling.
- No need for one-hot encoding — CatBoost handles categoricals internally.
- Combined with TTA ResNet embeddings + PCA.
- **Parameters**: iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=3.
- Categorical NaN values replaced with "Missing" for CatBoost strict mode.
- Model saved as `catboost_model.cbm`.
- **Submission**: `submission015.csv`

---

### Method 16: Iterative Refinement

- Iterative pseudo-labeling rounds.
- **Submission**: `submission016.csv`

---

### Method 17: EfficientNet Features

- Experimented with EfficientNet as backbone.
- **Submission**: `submission017.csv`

---

### Method 19: Vision Transformer (ViT) Features

- Experimented with ViT backbone for feature extraction.
- **Submission**: Generated predictions.

---

### Method 20: Spatial Target Encoding

- Added **K-Fold target encoding** for spatial categorical features:
  - `geolocation_name`
  - `country`
  - `region_economic_classification`
- Encodings created in OOF (out-of-fold) fashion to prevent target leakage.
- Test encodings derived from full training set mappings.
- Created `{col}_target_enc` features.

---

### Method 21: Visual KNN (Best Visual Model — RMSLE ≈ 0.2205)

This became the **best visual/multi-modal model** and a key component in later ensembles.

**Architecture**:
1. **Spatial Target Encoding** (from M20): OOF target encodings for `geolocation_name`, `country`, `region_economic_classification`.
2. **ResNet-18 TTA Embeddings**: 512-dim embeddings with 4× rotation averaging.
3. **Visual KNN Feature**: For each sample, find 50 nearest neighbors in embedding space (cosine similarity), compute mean log-target of neighbors → `visual_knn_cost` feature.
   - Train: OOF KNN (exclude self, use K+1 neighbors then skip index 0).
   - Test: Standard KNN against training embeddings.
4. **PCA**: 512 → 16 dimensions.
5. **LightGBM**: Using M09 best hyperparameters.
6. **Pseudo-Labeling**: Combined train + pseudo-labeled test data.
7. **Training**: 1,000 rounds on combined dataset.

**Key Innovation**: The `visual_knn_cost` feature leverages visual similarity to establish a "neighborhood average cost" — sites that look similar from satellite tend to have similar construction costs.

- **Leaderboard Score**: ~0.2205
- **Submission**: `submission021.csv`

---

### Methods 22–24: Bayesian Optimization, Stacking, Statistical Features

- **M22**: Bayesian optimization of hyperparameters.
- **M23**: Stacking ensemble of multiple base models.
- **M24**: Statistical/research-driven features (`is_remote_island`, `risk_index` = flood_risk + seismic_risk).

---

### Method 27: Hybrid Model (Full Feature Pipeline)

The most comprehensive single model combining ALL feature pipelines:

1. **Explicit Image Statistics** (from `image_features_train.csv`): 65 hand-crafted features.
2. **Research Features**: `is_remote_island`, `risk_index`, `quarter`.
3. **Spatial Target Encodings** (M20): OOF-encoded geographic features.
4. **Full ResNet-18 Embeddings**: 512 raw dimensions (NO PCA reduction — gave LightGBM full access).
5. **Visual KNN Feature**: `visual_knn_cost` (K=50, cosine similarity).
6. **Pseudo-Labeling**: Combined train + pseudo-labeled test.

**Validation**: GroupKFold (5 splits, grouped by `geolocation_name`) with pseudo-data added to each training fold.

**Hyperparameters** (regularized for high dimensionality):
```
learning_rate: 0.015
num_leaves: 60
feature_fraction: 0.7
bagging_fraction: 0.90
min_child_samples: 30
reg_alpha: 0.5
reg_lambda: 0.5
colsample_bytree: 0.7
```

- **Rounds**: 5,000 (early stopping 100) per fold; 1,500 for final retrain.
- **Submission**: `submission027.csv`

---

### Method 28: Recursive Feature Elimination (RFE)

- Applied RFE on top of M27's feature set.
- **Submission**: `submission028.csv`

---

### Ensemble Methods (Phase 1)

Multiple ensemble strategies were explored:

1. **Simple Weighted Average** (`ensemble.py`): Blends multiple submissions using weighted averaging in **log-space** (arithmetic average of log1p values → geometric mean in raw space).
   - Default: 60% M05 + 40% M04.

2. **Rank Ensemble** (`ensemble_rank.py`): Rank-based ensembling.

3. **Safety Ensemble** (`ensemble_safety.py`): Conservative blending for stable performance.

4. **Specialist Ensemble** (`ensemble_specialist.py`): Country-specific model blending.

5. **Optimized Ensemble** (`optimize_ensemble.py`): Grid search over blending weights.

Notable ensemble submissions:
| Submission | Combination | Notes |
|---|---|---|
| `sub_ens_kestav.csv` | Kestav's ensemble | **Teacher model** for pseudo-labeling (LB ~0.22046) |
| `sub_ens_kest_ram.csv` | Kestav + Ramcharan blend | Collaborative ensemble |
| `submission_ensemble.csv` | M04 + M05 | Initial ensemble |

---

## 6. Phase 2: AutoML & Graph Neural Networks (Methods 30–31)

### Method 30: AutoGluon Tabular

- Used **AutoGluon `TabularPredictor`** with `'best_quality'` presets.
- Time budget: **1 hour**.
- Log-transformed target: `log_cost = log1p(construction_cost_per_m2_usd)`.
- Metric: RMSE on log-target (equivalent to RMSLE).
- Combined tabular + extracted image features.
- **Submission**: `submission030.csv`

---

### Method 31: Graph Neural Network (GraphSAGE)

- **Architecture**: 2-layer GraphSAGE with ELU activations and dropout.
  ```
  SAGEConv(in_dim → 64) → ELU → Dropout(0.2) → SAGEConv(64 → 64) → ELU → Dropout(0.2) → Linear(64 → 1)
  ```
- **Graph Construction**: KNN graph (K=15) built from image feature similarity (visual neighbors).
  - Features: All Sentinel-2 and VIIRS statistics.
  - Distance: Ball Tree on StandardScaler-normalized features.
  - Undirected edges (bidirectional).
- **Validation**: GroupKFold (5 splits, grouped by `geolocation_name`).
- **Training**: 1,000 epochs per fold, Adam optimizer (lr=0.01, weight_decay=5e-4).
- **Inductive**: Trained on all nodes (train + test), but only train nodes have real labels.
- **Submission**: `submission031.csv`

---

## 7. Phase 3: Supervised Contrastive Learning (Method 32)

### Method 32: Supervised Contrastive (SupCon) + KNN

- **Stage 1 — Metric Learning**:
  - Pre-trained ResNet-18 used as encoder.
  - L2-normalized embeddings (512-dim).
  - **Online Triplet Loss** with hard mining:
    - Positive: Items with price difference < 0.1 (in log-space).
    - Negative: Items with price difference > 0.5 (in log-space).
    - Hard mining: Hardest positive (lowest similarity) & hardest negative (highest similarity).
    - Margin: 0.5.
  - **Training**: 30 epochs, Adam (lr=1e-4), batch_size=32.

- **Stage 2 — KNN Prediction**:
  - Fit KNN (K=50, cosine distance) on learned embeddings.
  - Weighted mean prediction: `weight = 1/distance`.

- **Submission**: `submission032.csv`

---

## 8. Phase 4: Deep Feature Selection & Domain Models (Methods 33–36)

### Method 33: Deep EDA & Feature Selection Analysis

A comprehensive statistical analysis (`analyze_m33.py`) was conducted to identify optimal feature subsets:

**Step 1 — Target Variable Analysis**:
- Count: 1,024
- Raw Skew: 0.0402, Log Skew: -0.2839
- Raw Kurtosis: -1.5161, Log Kurtosis: -1.7310

**Step 2 — Image Feature Redundancy (Hierarchical Clustering)**:
- 65 total image features clustered into ~22 groups using Ward linkage (distance threshold = 1.0).
- From each cluster, the feature with the highest Spearman correlation to the target was selected.
- **Reduced set: 23 features** (from 65).

Key clusters and selected representatives:
| Cluster | Size | Selected Representative |
|---|---|---|
| 9 | 3 (B1_mean, B2_mean, B4_mean…) | `s2_B4_mean` |
| 8 | 5 (B1_std, B2_std, B3_std…) | `s2_B1_std` |
| 4 | 5 (B6_mean, B7_mean, B8_mean…) | `s2_B9_mean` |
| 22 | 4 (viirs_mean, viirs_std, viirs_median…) | `viirs_max` |

**Step 3 — Feature Importance (Permutation Importance via Random Forest)**:

Top features by importance:
| Rank | Feature | Importance |
|---|---|---|
| 1 | `deflated_gdp_usd` | 1.8845 |
| 2 | `straight_distance_to_capital_km` | 0.0250 |
| 3 | `us_cpi` | 0.0127 |
| 4 | `viirs_max` | 0.0048 |
| 5 | `s2_B8A_median` | 0.0032 |
| 6 | `s2_B5_min` | 0.0024 |
| 7 | `seismic_hazard_zone_Moderate` | 0.0021 |
| 8 | `s2_B7_min` | 0.0020 |
| 9 | `s2_B9_max` | 0.0019 |
| 10 | `s2_B12_max` | 0.0018 |

**Final "Golden" Feature List** (Importance > 0.001): 21 features.

---

### Method 34: LightGBM with Golden Features + Optuna (50 Trials)

- Used the **21 Golden Features** from M33 analysis.
- **Optuna** hyperparameter search with **50 trials**.
- **GroupKFold CV** (5 splits, grouped by `geolocation_name`).
- Search space enforced **conservative regularization**:
  - `num_leaves`: 15–63, `max_depth`: 3–10, `min_child_samples`: 20–100.
- **Final Model**: 5-fold averaged predictions on test set.
- **Submission**: `submission034.csv`

---

### Method 35: Domain Expert Features (6 Features Only)

A minimal **6-feature domain model** designed for maximum interpretability and minimal overfitting:

**Domain Features** (`domain_features.json`):
```json
["deflated_gdp_usd", "viirs_mean", "s2_B11_mean", "s2_B8A_mean", "s2_B1_mean", "straight_distance_to_capital_km"]
```

**Rationale**:
- `deflated_gdp_usd`: Primary economic driver (>98% of cost variance from country-level GDP).
- `viirs_mean`: Nighttime light as urbanization proxy.
- `s2_B11_mean` / `s2_B8A_mean` / `s2_B1_mean`: Key spectral bands reflecting land use.
- `straight_distance_to_capital_km`: Logistics premium/accessibility.

**Training**: Optuna (20 trials) + GroupKFold(5) + LightGBM.
- **Leaderboard Score**: ~0.2291
- **Submission**: `submission035.csv`

---

### Method 36: Refined Feature Set

- An intermediate refinement between Golden (21) and Domain (6) feature sets.
- **Submission**: `submission036.csv`

---

## 9. Phase 5: Blending, Stacking & Final Optimization (Methods 37–48)

### Method 37: M21 + M35 Blend

- **Components**: 75% M21 (Visual) + 25% M35 (Scalar Domain).
- **Blending**: Arithmetic average in log-space (geometric mean in raw space).
- Correlation between M21 and M35: Checked to ensure sufficient diversity.
- **Submission**: `submission037.csv`

---

### Method 38: Pseudo-Labeling with Domain Features (RMSLE ≈ 0.2198)

A critical breakthrough — combining **pseudo-labeling** with the lean domain feature set:

- **Teacher**: `sub_ens_kestav.csv` (collaborative ensemble, LB ~0.22046).
- **Student Features**: 6 domain features from `domain_features.json`.
- **Training**: Combined train + pseudo-labeled test (~2,048 samples).
- **Optuna**: 20 trials for hyperparameter tuning.
- **Final Model**: Trained on ALL combined data (train + pseudo).
- **Leaderboard Score**: **~0.2198** (New personal best at the time).
- **Submission**: `submission038.csv`

---

### Method 39: Three-Way Blend (RMSLE ≈ 0.2191)

- **Strategy**: Blend the student (M38) with the most diverse strong model.
- **Components evaluated**: M38 (Student, 0.2198), M21 (Visual, 0.2205), Kestav (Teacher, 0.22046).
- **Correlation Analysis**:
  - M38 vs M21: Checked.
  - M38 vs Kestav: Checked.
  - Selected the partner with **lowest correlation** to M38.
- **Weights**: 60% M38 + 40% selected partner.
- **Leaderboard Score**: **~0.2191** (Improvement!).
- **Submission**: `submission039.csv`

---

### Method 40: Genetic Feature Selection (Evolution) (RMSLE ≈ 0.2196)

**Genetic Algorithm (GA)** for automated feature selection:

**GA Configuration**:
| Parameter | Value |
|---|---|
| Population Size | 20 |
| Generations | 15 |
| Mutation Rate | 0.1 |
| Elitism | 2 (top 2 preserved) |
| Selection | Tournament (size 3) |
| Crossover | Uniform |

**Evaluation Function**: 5-Fold KFold CV using fast LightGBM (300 rounds, 16 leaves, max_depth=6).

**Search Space**: All numeric features from tabular + image features (including pseudo-labeled test data for expanded training).

**Domain Expert Seeding**: The domain expert feature set was injected as individual #0 in the initial population as a strong baseline.

**Evolved Feature Set** (`evolved_features.json` — 28 features):
```json
["year", "deflated_gdp_usd", "us_cpi", "straight_distance_to_capital_km",
 "s2_B1_median", "s2_B1_min", "s2_B2_std", "s2_B2_max", "s2_B3_max",
 "s2_B4_std", "s2_B5_min", "s2_B6_mean", "s2_B6_median", "s2_B7_std",
 "s2_B8_mean", "s2_B8_min", "s2_B8A_mean", "s2_B8A_std", "s2_B8A_median",
 "s2_B9_max", "s2_B12_mean", "s2_B12_median", "s2_B12_max",
 "viirs_mean", "viirs_std", "viirs_median", "viirs_min", "viirs_max"]
```

- **Final Model**: 2,000 rounds on all data (train + pseudo).
- **Leaderboard Score**: ~0.2196.
- **Submission**: `submission040.csv`

---

### Method 41: M40 + M21 Blend

- **Components**: 60% M40 (Evolved Scalar) + 40% M21 (Visual).
- **Correlation**: M40 vs M21 checked for diversity.
- **Rationale**: M40 captures scalar/statistical patterns, M21 captures visual patterns (roof texture, urbanization).
- **Submission**: `submission041.csv`

---

### Method 42: Stacking Meta-Model (Ridge Regression)

**Architecture**: Two-level stacking:
- **Level 0 (Base Models)**: M21, M35, M40 — each generating OOF predictions.
- **Level 1 (Meta-Model)**: Ridge regression (α=0.1) combining base model predictions.

**OOF Generation** (`gen_oofs.py`):
1. **M21 OOF**: KNN (K=50, cosine) on ResNet-18 embeddings, cross-validated.
2. **M35 OOF**: LightGBM on 6 domain features, cross-validated.
3. **M40 OOF**: LightGBM on 28 evolved features, cross-validated.

**Stacking CV**: 5-Fold on meta-features → Ridge learns optimal combination weights.

**Learned Weights**: Coefficients for each base model printed and analyzed.

- **Submission**: `submission042.csv`

---

### Method 43: Conservative M39+M40 Blend

- **Components**: 80% M39 (champion at 0.2191) + 20% M40 (evolved scalar at 0.2196).
- **Rationale**: M39 is significantly better → don't risk ruining it. Just "nudge" with diverse M40 signal.
- **Submission**: `submission043.csv`

---

### Methods 44–45: Parabolic Weight Optimization

**Observation**: Multiple blend weights (M39+M40) at different ratios yield different LB scores. By fitting a **quadratic (parabolic) curve** to these data points, we can find the mathematically optimal blending weight.

**Data Points Used**:
| Weight (w for M40) | Score (RMSLE) |
|---|---|
| 0.0000 | 0.21912744 (M39) |
| 0.2000 | 0.21901124 (M43) |
| 0.2957 | 0.21899198 (M45 — then-best) |
| 1.0000 | 0.21957309 (M40) |

**Method**: `np.polyfit(weights, scores, 2)` → Quadratic curve → Derivative = 0 → Optimal w.

**Result**: The parabola confirmed w ≈ 0.2957 as near-optimal, with predicted score of ~0.21899.

- **Submission**: `submission044.csv`, `submission045.csv` (best: **0.21899**)

---

### Method 46: Bias Correction (Post-Processing)

- Measured systematic bias from M40 OOF predictions: `Bias = Pred - Target = -0.001577` (slight underprediction).
- Applied additive correction in log-space: `corrected_log = log_pred - bias`.
- **Submission**: `submission046.csv`

---

### Method 47: Recursive Student (Self-Distillation Attempt)

- Used M45 (champion submission) as pseudo-labels.
- Trained a new LightGBM student model on combined train + pseudo data.
- Used evolved features + all raw tabular columns.
- Categorical features handled with `'auto'` detection.
- **Outcome**: Did NOT improve over M45 — recursive distillation reached diminishing returns.
- **Submission**: `submission047.csv`

---

### Method 48: Final Parabola-Optimized Blend

- Re-ran parabolic optimization with all 4 data points.
- Generated final blend with mathematically optimal weight.
- **Submission**: `submission048.csv`

---

## 10. Results Summary & Leaderboard Scores

### RMSLE Progression (Key Submissions)

| Submission | Method | Description | RMSLE (LB) |
|---|---|---|---|
| `submission.csv` | M01 | Baseline LightGBM | — |
| `submission003.csv` | M03 | + HandCrafted Image Stats | — |
| `submission005.csv` | M05 | + ResNet-18 Embeddings | — |
| `submission007.csv` | M07 | + 5-Fold CV | — |
| `submission009.csv` | M09 | + Optuna Tuning | — |
| `submission010.csv` | M10 | End-to-End Deep Learning | — |
| `submission013.csv` | M13 | + Pseudo-Labeling | — |
| `submission015.csv` | M15 | CatBoost | — |
| `submission021.csv` | M21 | **Visual KNN Champion** | **~0.2205** |
| `sub_ens_kestav.csv` | Ensemble | Collaborative Ensemble (Teacher) | **~0.22046** |
| `submission030.csv` | M30 | AutoGluon | — |
| `submission031.csv` | M31 | Graph Neural Network | — |
| `submission032.csv` | M32 | SupCon + KNN | — |
| `submission034.csv` | M34 | Golden Features + Optuna | — |
| `submission035.csv` | M35 | Domain Expert (6 features) | **~0.2291** |
| `submission037.csv` | M37 | M21+M35 Blend | — |
| `submission038.csv` | M38 | **Pseudo + Domain** | **~0.2198** |
| `submission039.csv` | M39 | **Three-Way Blend** | **~0.2191** |
| `submission040.csv` | M40 | **Genetic Evolution (28 feats)** | **~0.2196** |
| `submission041.csv` | M41 | M40+M21 Blend | — |
| `submission042.csv` | M42 | Ridge Stacking | — |
| `submission043.csv` | M43 | M39+M40 Conservative Blend | **~0.2190** |
| `submission044.csv` | M44 | Parabola-Optimized Blend | — |
| **`submission045.csv`** | **M45** | **Parabola-Optimized (Best)** | **~0.21899** |
| `submission046.csv` | M46 | Bias-Corrected M45 | — |
| `submission047.csv` | M47 | Recursive Student (Failed) | — |
| `submission048.csv` | M48 | Final Parabola Optimization | — |

### Per-Country Model Performance (Country-Split Pipeline)

| Country | Model | CV RMSLE | CV R² |
|---|---|---|---|
| Japan | Linear Regression | — | — |
| Philippines | Linear Regression | — | — |
| Japan | Random Forest | — | — |
| Philippines | Random Forest | — | — |
| **Japan** | **LightGBM (Tuned)** | **0.1011** | — |
| **Philippines** | **LightGBM (Tuned)** | **0.2387** | — |

**Key per-country LightGBM hyperparameters** (from Optuna):
- **Japan**: lr=0.0156, num_leaves=22, max_depth=5, min_child_samples=11, subsample=0.80, colsample=0.83.
- **Philippines**: lr=0.0210, num_leaves=136, max_depth=13, min_child_samples=72, subsample=0.74, colsample=0.93.

---

## 11. Feature Lists Used

### Domain Features (6 features — `domain_features.json`)
```
deflated_gdp_usd, viirs_mean, s2_B11_mean, s2_B8A_mean, s2_B1_mean, straight_distance_to_capital_km
```

### Golden Features (21 features — `golden_features.json`)
```
deflated_gdp_usd, straight_distance_to_capital_km, us_cpi, viirs_max, s2_B8A_median,
s2_B5_min, seismic_hazard_zone, s2_B7_min, s2_B9_max, s2_B12_max, s2_B9_mean,
s2_B11_max, s2_B4_median, s2_B11_std, s2_B1_max, s2_B8A_max, s2_B12_median,
s2_B11_mean, s2_B6_std, s2_B3_min, viirs_min
```

### Evolved Features (28 features — `evolved_features.json`)
```
year, deflated_gdp_usd, us_cpi, straight_distance_to_capital_km,
s2_B1_median, s2_B1_min, s2_B2_std, s2_B2_max, s2_B3_max,
s2_B4_std, s2_B5_min, s2_B6_mean, s2_B6_median, s2_B7_std,
s2_B8_mean, s2_B8_min, s2_B8A_mean, s2_B8A_std, s2_B8A_median,
s2_B9_max, s2_B12_mean, s2_B12_median, s2_B12_max,
viirs_mean, viirs_std, viirs_median, viirs_min, viirs_max
```

---

## 12. Key Findings & Insights

### 12.1 What Worked

1. **Country Separation**: The bimodal distribution means treating Japan/Philippines separately (or using country as a strong feature) is essential.

2. **Log Transform**: `log1p` target transformation is mandatory for RMSLE optimization — reduces skewness from 0.20 to -0.05.

3. **deflated_gdp_usd is King**: Importance = 1.8845 — orders of magnitude above all other features. This single feature explains >98% of the variance (effectively encodes country + economic zone).

4. **Visual KNN**: The `visual_knn_cost` feature (average price of 50 most visually similar sites) is the strongest engineered feature. Sites that look similar from satellite DO have similar costs.

5. **Pseudo-Labeling**: Self-training with teacher labels consistently improved scores by expanding the effective training set from 1,024 to ~2,048 samples.

6. **Diverse Ensembling**: The key to breaking 0.22 was blending models with different information sources:
   - Scalar models (M35/M38/M40): Capture economic/spectral statistics.
   - Visual models (M21): Capture spatial patterns visible from imagery.
   - Low correlation between these model types enables effective blending.

7. **Genetic Feature Selection**: Evolved feature sets (28 features) outperformed both hand-crafted domain features (6) and golden features (21), discovering non-obvious combinations.

8. **Parabolic Weight Optimization**: Mathematically fitting the LB score curve to find optimal blend weights yielded the final best score.

### 12.2 What Didn't Work

1. **End-to-End Deep Learning (M10)**: Small dataset (1,024 images) is insufficient for stable CNN fine-tuning compared to LightGBM with pre-extracted features.

2. **Graph Neural Networks (M31)**: GraphSAGE on visual KNN graphs added complexity without clear gains over simpler approaches.

3. **Supervised Contrastive Learning (M32)**: Metric learning + KNN was inferior to simply using pre-trained ImageNet features.

4. **Recursive Self-Distillation (M47)**: Iterating pseudo-labeling beyond one round did not improve — diminishing returns.

5. **Bias Correction (M46)**: Additive bias correction in log-space didn't reliably improve LB scores.

6. **Some dropping of features**: Dropping `access_to_highway` (always present) was good, but dropping `year` in cleaning hurt the evolved model which found `year` useful (RMSLE ~0.2196 with it).

### 12.3 Critical Technical Lessons

- **GroupKFold > KFold**: Using `geolocation_name` as groups prevents data leakage between same-location samples across folds.
- **TTA Averaging**: 4× rotation TTA for satellite embeddings reduces noise and improves feature stability.
- **Log-space blending**: Arithmetic average in log-space (geometric mean in raw space) is mathematically optimal for RMSLE.
- **Feature fraction**: Reducing `feature_fraction` and `colsample_bytree` when using 500+ features prevents overfitting.

---

## 13. Future Plans & Potential Improvements

### 13.1 Modeling Ideas Not Yet Explored

1. **XGBoost / LightGBM with DART**: DART boosting mode can reduce overfitting by randomly dropping trees.
2. **Neural Network Tabular Models**: TabNet, FT-Transformer for tabular-first learning.
3. **Multi-Task Learning**: Jointly predict cost + country (auxiliary classification task).
4. **Advanced Image Models**: DINOv2, CLIP embeddings for richer visual features.
5. **Temporal Models**: Exploit the 2019–2024 time series structure with time-aware cross-validation.

### 13.2 Feature Engineering

1. **NDVI / NDBI Indices**: Compute vegetation and built-up indices from Sentinel-2 bands.
2. **Spatial Features**: Latitude/longitude, distance to nearest urban center, population density.
3. **Cross-feature Interactions**: GDP × distance, CPI × spectral features.

### 13.3 Ensemble Strategy

1. **Bayesian Model Combination**: Optimize ensemble weights using Bayesian optimization instead of grid search or parabola fitting.
2. **More Diverse Base Models**: Include CatBoost, XGBoost, and Random Forest as additional stacking candidates.
3. **Country-Specific Ensembles**: Different blend weights for Japan vs Philippines predictions.

---

## 14. Project Structure

```
construction/
├── dataset/
│   ├── train_tabular.csv               # Raw training tabular data
│   ├── train_composite/                # Sentinel-2/VIIRS composite GeoTIFFs (train)
│   ├── train_png/                      # PNG conversions of composite images
│   ├── image_features_train.csv        # 65 hand-crafted image features (train)
│   ├── embeddings_resnet18_train.npy   # ResNet-18 embeddings (512-dim, train)
│   ├── Research.md                     # Research paper / EDA findings
│   └── top_correlations_matrix.csv     # Correlation analysis
│
├── evaluation_dataset/
│   ├── evaluation_tabular_no_target.csv # Test tabular data (no labels)
│   ├── evaluation_composite/            # Test composite GeoTIFFs
│   ├── test_png/                        # Test PNG images
│   ├── image_features_test.csv          # Image features (test)
│   └── embeddings_resnet18_test.npy     # ResNet-18 embeddings (test)
│
├── data_clean/
│   ├── clean_data.py                   # Data cleaning pipeline (split + encode + impute)
│   ├── verify_data.py                  # Post-cleaning verification & plots
│   ├── analyze_ph_capital.py           # Philippines capital redundancy analysis
│   ├── verify_log.txt                  # Verification output log
│   ├── train_japan.csv                 # Cleaned Japan dataset
│   └── train_philippines.csv           # Cleaned Philippines dataset
│
├── EDA_try/
│   ├── EDA_report.py                   # YData Profiling report generator
│   ├── EDA_Report.html                 # Full EDA report (277MB!)
│   ├── comp.py, geo_loc.py, risk.py    # Specialized EDA scripts
│   └── *.png                           # EDA visualization plots
│
├── new/                                # Kestav's branch (Japan/Philippines notebooks)
│   ├── japan.ipynb, phil.ipynb          # Per-country EDA notebooks
│   ├── main.ipynb                      # Main analysis notebook
│   ├── _gen_country_notebooks.py       # Notebook generator
│   └── saved_models*/                  # Saved model artifacts
│
├── 01a_linear_regression/              # Linear Regression (per-country)
│   ├── train_lr.py                     # Training script
│   └── predict.py                      # Prediction script
│
├── 02_random_forest/                   # Random Forest (per-country)
│   ├── train_rf.py                     # Training script
│   └── predict.py                      # Prediction script
│
├── 03_lightgbm/                        # LightGBM (per-country, Optuna-tuned)
│   ├── train_lgbm.py                   # Training with Optuna-tuned params
│   ├── tune_lgbm_japan.py              # Japan-specific Optuna tuning
│   ├── tune_lgbm_philippines.py        # Philippines-specific Optuna tuning
│   └── predict_hybrid.py              # Hybrid prediction
│
├── phase1_archive/                     # Methods 01–29 (core development)
│   ├── train.py                        # M01: Baseline LightGBM
│   ├── train_03.py                     # M03: + Image Statistics
│   ├── train_05.py                     # M05: + CNN Embeddings (PCA)
│   ├── train_07_cv.py                  # M07: 5-Fold CV
│   ├── train_08_tta.py                 # M08: TTA Embeddings
│   ├── train_09_optuna.py              # M09: Optuna Tuning (25 trials)
│   ├── train_10_dl.py                  # M10: Deep Learning (ResNet fine-tune)
│   ├── train_13_pseudo.py              # M13: Pseudo-Labeling
│   ├── train_14_deep_pseudo.py         # M14: Deep Pseudo-Labeling
│   ├── train_15_catboost.py            # M15: CatBoost
│   ├── train_16_iterative.py           # M16: Iterative Refinement
│   ├── train_17_efficient.py           # M17: EfficientNet
│   ├── train_19_vit.py                 # M19: Vision Transformer
│   ├── train_20_spatial.py             # M20: Spatial Target Encoding
│   ├── train_21_visknn.py              # M21: Visual KNN (Best Visual Model)
│   ├── train_22_bayes.py               # M22: Bayesian Optimization
│   ├── train_23_stacking.py            # M23: Stacking Ensemble
│   ├── train_24_stats.py               # M24: Statistical Features
│   ├── train_27_hybrid.py              # M27: Full Hybrid Pipeline
│   ├── train_28_rfe.py                 # M28: Recursive Feature Elimination
│   ├── ensemble*.py                    # Various ensemble scripts
│   ├── extract_features.py             # Image feature extraction pipeline
│   ├── predict_*.py                    # Prediction scripts for each method
│   └── submission*.csv                 # All Phase 1 submissions (001–029)
│
├── phase2_archive/                     # Methods 30–31
│   ├── train_30_autogluon.py           # AutoGluon TabularPredictor
│   └── train_31_gnn.py                 # GraphSAGE Neural Network
│
├── phase3_archive/                     # Method 32
│   └── train_32_supcon.py              # Supervised Contrastive Learning
│
├── phase4_archive/                     # Methods 33–36
│   ├── analyze_m33.py                  # Deep EDA & Feature Selection
│   ├── feature_selection_report.md     # Feature importance analysis
│   ├── golden_features.json            # 21 selected features
│   ├── train_34_lgbm.py               # Golden Features + Optuna
│   ├── train_35_domain.py              # Domain Expert features (6)
│   └── train_36_refined.py             # Refined feature set
│
├── domain_features.json                # 6 domain expert features
├── evolved_features.json               # 28 genetically evolved features
├── blend_m37.py                        # M37: M21+M35 Blend
├── train_38_pseudo.py                  # M38: Pseudo + Domain
├── blend_m39.py                        # M39: Three-Way Blend
├── train_40_evolution.py               # M40: Genetic Feature Selection
├── blend_m41.py                        # M41: M40+M21 Blend
├── train_42_stacking.py                # M42: Ridge Stacking
├── blend_m43.py                        # M43: Conservative Blend
├── optimize_parabola.py                # M44: Parabolic Optimization
├── optimize_parabola_v2.py             # M45/M48: Refined Parabola
├── correct_bias.py                     # M46: Bias Correction
├── train_47_recursive.py               # M47: Recursive Student
├── gen_embeddings.py                   # Embedding generation utility
├── gen_oofs.py                         # OOF prediction generation
├── oofs_train.csv / oofs_test.csv      # Generated OOF predictions
├── submission037–048.csv               # Phase 5 submissions
├── plots/                              # All visualization outputs
│   ├── post_split_eda/                 # Post-cleaning verification plots
│   ├── 01a_linear_regression/          # LR performance plots
│   ├── 02_random_forest/              # RF feature importance plots
│   └── 03_lightgbm/                   # LightGBM performance plots
├── LightGBM/                           # Built LightGBM from source (CUDA support)
└── requirements.txt                    # Python dependencies
```

---

## Appendix A: Technical Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.x |
| **ML Framework** | LightGBM (CUDA), CatBoost, scikit-learn |
| **Deep Learning** | PyTorch + torchvision (ResNet-18, EfficientNet, ViT) |
| **GNN** | PyTorch Geometric (GraphSAGE) |
| **AutoML** | AutoGluon, Optuna |
| **Image Processing** | rasterio, Pillow (PIL) |
| **Feature Engineering** | PCA (scikit-learn), KNN (scikit-learn) |
| **EDA** | ydata-profiling, seaborn, matplotlib |
| **Computation** | CUDA GPU acceleration |

---

## Appendix B: Blending Weight Evolution

The final champion submission (M45) was achieved through systematic optimization of the M39+M40 blending weight:

```
M39 Only (w=0.0):           0.21912744
M43 (w=0.2):                0.21901124
M45 (w=0.2957):             0.21899198  ← BEST
M40 Only (w=1.0):           0.21957309
```

The parabolic fit `aw² + bw + c` confirmed that the optimal weight lies near w ≈ 0.30, with the minimum predicted score matching the empirical M45 result almost exactly.

---

*Report generated on 2026-03-31. Total Methods Explored: 48+. Total Submissions: 55+.*
