## Plan: Best-Practices ML Regression Pipeline

**TL;DR** — Build a modular, reusable regression pipeline in main.ipynb targeting `construction_cost_per_m2_usd`. The notebook will cover: EDA & cleaning → feature engineering → preprocessing (scaling numericals, encoding categoricals with target encoding for high-cardinality) → training 9 models (Ridge, Lasso, ElasticNet, SVR, RF, GBR, XGBoost, LightGBM, CatBoost) with cross-validation and hyperparameter tuning via `RandomizedSearchCV` → comparison dashboard with metrics, residual plots, learning curves, and feature importances.

---

**Steps**

### Phase 1: EDA & Cleaning (cells 1–5)

1. **Imports cell** — Import pandas, numpy, matplotlib, seaborn, sklearn, xgboost, lightgbm, catboost, and warnings filter. Detect CUDA availability via `torch.cuda.is_available()` or device check and print GPU info.

2. **Load & inspect** — Read train_tabular.csv, display `.shape`, `.info()`, `.describe()`, `.isnull().sum()`, `.duplicated().sum()`.

3. **Drop useless columns** — Remove `data_id`, `sentinel2_tiff_file_name`, `viirs_tiff_file_name`.

4. **EDA visualizations**:
   - Target distribution histogram + boxplot
   - Correlation heatmap for numerical features
   - Box/violin plots of target by key categoricals (`country`, `developed_country`, `region_economic_classification`, `seismic_hazard_zone`)
   - Pairplot of numerical features vs target
   - Value counts of all categorical columns

5. **Feature engineering** — Extract `quarter` (1–4) from `quarter_label`, drop `quarter_label`. Convert `year` to int if needed.

### Phase 2: Preprocessing (cells 6–7)

6. **Define column groups**:
   - **Numerical**: `year`, `quarter`, `deflated_gdp_usd`, `us_cpi`, `straight_distance_to_capital_km`
   - **Binary categorical** (Yes/No → 0/1): `developed_country`, `landlocked`, `access_to_airport`, `access_to_port`, `access_to_highway`, `access_to_railway`, `flood_risk_class`
   - **Low-cardinality categorical** (one-hot): `country`, `region_economic_classification`, `seismic_hazard_zone`, `tropical_cyclone_wind_risk`, `tornadoes_wind_risk`, `koppen_climate_zone`
   - **High-cardinality categorical** (target encoding): `geolocation_name`

7. **Build `ColumnTransformer`**:
   - `StandardScaler` for numerical features
   - `OrdinalEncoder(categories=[["No","Yes"]])` for binary features
   - `OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist')` for low-cardinality categoricals
   - `TargetEncoder()` (sklearn 1.3+) for `geolocation_name`

### Phase 3: Model Training & Tuning (cells 8–10)

8. **Define model dictionary** — Each entry has a `Pipeline(preprocessor, model)` and a `param_grid` dict for `RandomizedSearchCV`:
   - **Ridge** — `alpha` range
   - **Lasso** — `alpha` range
   - **ElasticNet** — `alpha`, `l1_ratio`
   - **SVR** — `C`, `epsilon`, `kernel`
   - **RandomForestRegressor** — `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
   - **GradientBoostingRegressor** — `n_estimators`, `max_depth`, `learning_rate`, `subsample`
   - **XGBRegressor** — `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, **`device='cuda'`** (GPU-accelerated)
   - **LGBMRegressor** — `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `subsample`, **`device='gpu'`** (GPU-accelerated)
   - **CatBoostRegressor** — `iterations`, `depth`, `learning_rate`, **`task_type='GPU'`** (silent mode, GPU-accelerated)

9. **Training loop** — For each model:
   - Run `RandomizedSearchCV` with `cv=5` (KFold), `scoring='neg_mean_squared_error'`, `n_iter=30`, `random_state=42`
   - Store best estimator, best params, and CV results
   - Print progress

10. **Results comparison table** — DataFrame with columns: Model, Best RMSE (CV), Best MAE (CV), Best R² (CV), Best Params. Sort by RMSE ascending.

### Phase 4: Evaluation & Visualization (cells 11–20)

All plots are saved as high-res PNGs to `plots/` directory via `plt.savefig()` **and** displayed inline.

11. **Cross-validation metrics bar chart** — Grouped bar chart comparing RMSE, MAE, R² across all 9 models. → `plots/01_model_comparison_metrics.png`

12. **Hyperparameter tuning curves** — For each model, plot `RandomizedSearchCV` validation score vs each hyperparameter (scatter plots with trend line). Shows how each hyperparameter affects performance. → `plots/02_hyperparam_<ModelName>_<param>.png`

13. **Validation curve plots** — For top 3 models, use `sklearn.model_selection.validation_curve()` to plot train vs validation score as a function of the most important hyperparameter (e.g. `alpha` for Ridge, `n_estimators` for RF, `max_depth` for XGBoost). → `plots/03_validation_curve_<ModelName>.png`

14. **Learning curves** — For top 3 models: `learning_curve()` plot showing train vs validation score over training set size. Diagnoses overfitting/underfitting. → `plots/04_learning_curve_<ModelName>.png`

15. **Residual plots** — For top 3 models:
   - Predicted vs Actual scatter (with 45° ideal line) → `plots/05_pred_vs_actual_<ModelName>.png`
   - Residuals vs Predicted scatter (check homoscedasticity) → `plots/06_residuals_vs_predicted_<ModelName>.png`
   - Residual distribution histogram + KDE (check normality) → `plots/07_residual_distribution_<ModelName>.png`

16. **Feature importance** — For tree-based models (RF, GBR, XGBoost, LightGBM, CatBoost): horizontal bar chart of top 15 feature importances. For linear models (Ridge, Lasso, ElasticNet): coefficient magnitude bar chart. → `plots/08_feature_importance_<ModelName>.png`

17. **Cumulative error distribution curve** — For all models on one plot: X-axis = absolute error threshold, Y-axis = % of predictions within that error. The regression equivalent of an ROC curve — shows prediction quality at different tolerance levels. → `plots/09_cumulative_error_distribution.png`

18. **Prediction error plot** — For top 3 models: `PredictionError` display from sklearn (actual vs predicted with quantile lines). → `plots/10_prediction_error_<ModelName>.png`

19. **Cross-validation fold-wise box plots** — For all models: box plots showing the distribution of scores across the 5 CV folds (reveals variance/stability). → `plots/11_cv_fold_scores_boxplot.png`

20. **Correlation of model predictions** — Heatmap showing how correlated each model's predictions are with each other (useful for ensembling decisions). → `plots/12_model_prediction_correlation.png`

### Phase 5: Model Persistence & Reusability (cells 21–22)

21. **Save ALL trained models** — After the training loop, use `joblib.dump` to save every best estimator (full pipeline including preprocessor) to a `saved_models/` directory:
   - `saved_models/Ridge.joblib`
   - `saved_models/Lasso.joblib`
   - `saved_models/ElasticNet.joblib`
   - `saved_models/SVR.joblib`
   - `saved_models/RandomForest.joblib`
   - `saved_models/GradientBoosting.joblib`
   - `saved_models/XGBoost.joblib`
   - `saved_models/LightGBM.joblib`
   - `saved_models/CatBoost.joblib`
   - Also save `saved_models/results_summary.csv` (the comparison table with all CV metrics and best params).
   - This means **you never need to retrain** — just load a model and call `.predict()`.

22. **Load & reuse cell** — A standalone utility cell that:
   - Loads any saved model via `joblib.load('saved_models/<ModelName>.joblib')`
   - Demonstrates prediction on new raw data (reads `evaluation_tabular_no_target.csv`, applies same column drops + quarter extraction, then calls `pipeline.predict()`)
   - This cell works independently — you can restart the kernel and run only this cell to get predictions without retraining.

---

**Verification**
- All cells run top-to-bottom without errors
- CV scores are reproducible (`random_state=42` everywhere)
- Confirm no data leakage: target encoding uses CV-internal folds only (sklearn's `TargetEncoder` handles this)
- Check that `df.isnull().sum()` is zero after preprocessing
- Verify model comparison table shows sensible R² values (expect 0.7+ given strong country signal)
- Verify all 9 `.joblib` files exist in `saved_models/` after training
- Verify the reload cell works in a fresh kernel (no retraining needed)

**Decisions**
- Target encoding for `geolocation_name` over one-hot (avoids ~100+ sparse columns on 1025 rows)
- `RandomizedSearchCV` over `GridSearchCV` (faster with 9 models, still good coverage with `n_iter=30`)
- `StandardScaler` for numericals (needed for Ridge/Lasso/SVR; tree models are invariant but no harm)
- CUDA/GPU acceleration for XGBoost (`device='cuda'`), LightGBM (`device='gpu'`), and CatBoost (`task_type='GPU'`) to speed up training
- Cumulative error distribution curve instead of ROC (ROC is classification-only; this is the regression equivalent for diagnostic quality)
- Extract quarter from `quarter_label` to avoid redundancy with `year`
