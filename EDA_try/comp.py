import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TABULAR_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_tabular.csv")
df = pd.read_csv(TABULAR_PATH)
# ---------------------------------------------------------
# 1. Split the Data
# ---------------------------------------------------------
df_ph = df[df['country'] == 'Philippines'].copy()
df_jp = df[df['country'] == 'Japan'].copy()

# ---------------------------------------------------------
# 2. Comparative Correlation Analysis
# ---------------------------------------------------------
corr_ph = (
    df_ph
    .corr(numeric_only=True)['construction_cost_per_m2_usd']
    .sort_values(ascending=False)
)

corr_jp = (
    df_jp
    .corr(numeric_only=True)['construction_cost_per_m2_usd']
    .sort_values(ascending=False)
)

# Combine into a single DataFrame (horizontal join)
comparison = pd.DataFrame({
    'Philippines Correlation': corr_ph,
    'Japan Correlation': corr_jp
})

# ---------------------------------------------------------
# 3. Visualize the Difference (SAVE heatmap)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(
    comparison,
    annot=True,
    cmap='coolwarm',
    center=0
)
plt.title('Do Cost Drivers Change by Country?')

plt.savefig(
    "cost_driver_correlation_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ---------------------------------------------------------
# 4. Check for "Local" Outliers (SAVE boxplots)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(
    y=df_ph['construction_cost_per_m2_usd'],
    ax=axes[0],
    color='skyblue'
)
axes[0].set_title('Philippines Cost Distribution')

sns.boxplot(
    y=df_jp['construction_cost_per_m2_usd'],
    ax=axes[1],
    color='orange'
)
axes[1].set_title('Japan Cost Distribution')

plt.savefig(
    "construction_cost_distribution_ph_vs_jp.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
