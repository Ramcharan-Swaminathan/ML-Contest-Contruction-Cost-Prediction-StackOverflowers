## Plan: Country-Split ML Pipelines (Japan & Philippines)

**TL;DR** — Split the training data by country and create two separate notebooks (`japan.ipynb` and `phil.ipynb`), each with the full best-practices pipeline. By removing the country-level signal, models are forced to learn granular, local cost drivers (distance to capital, seismic risk, climate zone, etc.) — producing much more honest and usable per-country RMSE/MAE metrics.

---

**Why Split?**

1. **Honest error metrics** — The combined model's RMSE (~171) is acceptable for Japan (~$1,500–2,000 range, ~10% error) but disastrous for Philippines (~$200 range, ~85% error). Separate models will yield country-appropriate errors (expect Philippines RMSE ~$20–40, Japan RMSE ~$100–200).

2. **Unmasked local feature importance** — The combined model relies on macro-indicators (`us_cpi`, `deflated_gdp_usd`, `country`) because the country gap dominates. Split models will reveal the true impact of `straight_distance_to_capital_km`, `seismic_hazard_zone`, `tropical_cyclone_wind_risk`, `koppen_climate_zone`, etc.

3. **Eliminated multicollinearity** — Within a single country, `country`, `developed_country`, `region_economic_classification` are constants and can be dropped entirely, cleaning up the feature space.

---

**Steps**

### Common: Columns to Drop Per Country

These become constant (or near-constant) within each country and add no predictive value:

- `data_id` (identifier)
- `sentinel2_tiff_file_name`, `viirs_tiff_file_name` (file paths)
- `country` (constant — "Japan" or "Philippines")
- `developed_country` (constant — "Yes" for Japan, "No" for Philippines)
- `region_economic_classification` (constant — "High income" for Japan, "Lower-middle income" for Philippines)
- `landlocked` (constant — "No" for both countries)
- `quarter_label` (replaced by extracted `quarter`)

### Japan Notebook (`japan.ipynb`)

**Phase 1: EDA & Cleaning (cells 1–11)**
1. Imports & config — same as main.ipynb but with `COUNTRY = 'Japan'`, output dirs `plots_japan/`, `saved_models_japan/`
2. Load data — `df = pd.read_csv('train_tabular.csv')` then filter `df = df[df['country'] == 'Japan']`
3. Drop useless + constant columns — `data_id`, `sentinel2_tiff_file_name`, `viirs_tiff_file_name`, `country`, `developed_country`, `region_economic_classification`, `landlocked`
4. Inspect — `.shape`, `.info()`, `.describe()`, `.isnull().sum()`
5. Target distribution — histogram + boxplot (expect ~$1,400–2,600 range, roughly normal)
6. Correlation heatmap — numerical features only
7. Box/violin plots — target vs remaining categoricals: `seismic_hazard_zone`, `tropical_cyclone_wind_risk`, `tornadoes_wind_risk`, `koppen_climate_zone`, `flood_risk_class`, `access_to_*`
8. Pairplot — numerical features vs target
9. Value counts — all remaining categorical columns
10. Feature engineering — extract `quarter` from `quarter_label`, drop `quarter_label`
11. Check for any additional near-constant columns within Japan and drop them

**Phase 2: Preprocessing (cells 12–13)**
12. Define column groups (Japan-specific):
    - **Numerical**: `year`, `quarter`, `deflated_gdp_usd`, `us_cpi`, `straight_distance_to_capital_km`
    - **Binary categorical** (Yes/No → 0/1): `access_to_airport`, `access_to_port`, `access_to_highway`, `access_to_railway`, `flood_risk_class`
    - **Low-cardinality categorical** (one-hot): `seismic_hazard_zone`, `tropical_cyclone_wind_risk`, `tornadoes_wind_risk`, `koppen_climate_zone`
    - **High-cardinality categorical** (target encoding): `geolocation_name`
13. Build `ColumnTransformer` — same transformers as main but with Japan-specific column lists

**Phase 3: Model Training & Tuning (cells 14–16)**
14. Define 9 models + param grids — identical to main.ipynb (CUDA for XGBoost/CatBoost, CPU for LightGBM)
15. Resumable training loop — `RandomizedSearchCV` with 5-fold CV, 30 iterations
16. Results comparison table

**Phase 4: Visualization (cells 17–26)**
17. Model comparison bar chart → `plots_japan/01_model_comparison_metrics.png`
18. Hyperparameter tuning curves → `plots_japan/02_hyperparam_<Model>.png`
19. Validation curves (top 3) → `plots_japan/03_validation_curves_top3.png`
20. Learning curves (top 3) → `plots_japan/04_learning_curves_top3.png`
21. Residual analysis (top 3) → `plots_japan/05_residual_analysis_<Model>.png`
22. Feature importance → `plots_japan/08_feature_importance_*.png`
23. Cumulative error distribution → `plots_japan/09_cumulative_error_distribution.png`
24. Prediction error plot (top 3) → `plots_japan/10_prediction_error_top3.png`
25. CV fold-wise box plots → `plots_japan/11_cv_fold_scores_boxplot.png`
26. Model prediction correlation → `plots_japan/12_model_prediction_correlation.png`

**Phase 5: Persistence (cells 27–29)**
27. Save all 9 models → `saved_models_japan/<Model>.joblib`
28. Standalone reload + predict cell — filters eval set to Japan rows, applies same preprocessing, predicts
29. Summary of saved artifacts

### Philippines Notebook (`phil.ipynb`)

Identical structure to Japan notebook with:
- `COUNTRY = 'Philippines'`
- Output dirs: `plots_phil/`, `saved_models_phil/`
- Filter: `df = df[df['country'] == 'Philippines']`
- Same constant columns dropped
- Same 9 models, same hyperparameter grids, same CV setup
- All plots and models saved to `plots_phil/` and `saved_models_phil/`

### Inference / Submission Cell (in both notebooks)

The standalone reload cell in each notebook will:
1. Load `evaluation_tabular_no_target.csv`
2. Filter to the respective country
3. Apply same column drops + quarter extraction
4. Load the best saved model
5. Generate predictions
6. Save to `predictions_japan_<Model>.csv` or `predictions_phil_<Model>.csv`

A final merge step (can be in either notebook or main.ipynb) combines both country predictions into a single submission file.

---

**Key Differences from main.ipynb**

| Aspect | main.ipynb (combined) | japan.ipynb / phil.ipynb |
|---|---|---|
| Data | All 1025 rows | ~512 Japan / ~513 Philippines |
| Dropped columns | 3 (id, tiff files) | 7+ (id, tiff, country, developed_country, region_econ, landlocked, quarter_label) |
| Target range | $100–3,500 | Japan: $1,400–2,600 / Phil: $100–460 |
| RMSE interpretation | Misleading (mixes scales) | Honest per-country error |
| Feature importance | Dominated by country/GDP | Local drivers (distance, seismic, climate) |
| Multicollinearity | High (country↔GDP↔developed) | Eliminated |

---

**Verification**
- Both notebooks run top-to-bottom without errors
- Japan RMSE should be significantly lower than ~171 (the combined model's overall RMSE)
- Philippines RMSE should drop dramatically (expect ~$20–40 vs ~$171)
- Feature importances should show local features (distance, seismic, etc.) ranking higher than in the combined model
- All models saved to separate directories (`saved_models_japan/`, `saved_models_phil/`)
- All plots saved to separate directories (`plots_japan/`, `plots_phil/`)
- Standalone reload cells work independently in a fresh kernel
- Combined predictions from both notebooks match the full evaluation set row count
