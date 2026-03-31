import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TABULAR_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_tabular.csv")
df = pd.read_csv(TABULAR_PATH)

# ---------------------------------------------------------
# Split by country
# ---------------------------------------------------------
df_ph = df[df['country'] == 'Philippines'].copy()
df_jp = df[df['country'] == 'Japan'].copy()

# ---------------------------------------------------------
# Columns to fix (boolean-like)
# ---------------------------------------------------------
cols_to_fix = [
    'access_to_railway',
    'access_to_port',
    'access_to_airport',
    'access_to_highway',
    'flood_risk_class',
    'landlocked',
    'developed_country'
]

# Mapping for categorical booleans
bool_map = {
    'Yes': 1,
    'No': 0,
    'Y': 1,
    'N': 0,
    True: 1,
    False: 0
}

# Create copies for correlation
df_ph_corr = df_ph.copy()
df_jp_corr = df_jp.copy()

# ---------------------------------------------------------
# Safe conversion
# ---------------------------------------------------------
for col in cols_to_fix:
    if col in df_ph_corr.columns:
        df_ph_corr[col] = df_ph_corr[col].map(bool_map)

    if col in df_jp_corr.columns:
        df_jp_corr[col] = df_jp_corr[col].map(bool_map)

# ---------------------------------------------------------
# Correlation with target
# ---------------------------------------------------------
target = 'construction_cost_per_m2_usd'

corrs_ph = (
    df_ph_corr
    .corr(numeric_only=True)[target]
    .drop(target)
)

corrs_jp = (
    df_jp_corr
    .corr(numeric_only=True)[target]
    .drop(target)
)

corr_diff = (
    pd.DataFrame({
        'Philippines': corrs_ph,
        'Japan': corrs_jp
    })
    .sort_values(by='Japan', ascending=False)
)

# ---------------------------------------------------------
# Save heatmap
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_diff,
    annot=True,
    cmap='RdBu',
    center=0,
    fmt=".2f"
)
plt.title('Which Infrastructure Matters in Which Country?')

plt.savefig(
    "infrastructure_correlation_ph_vs_jp.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
