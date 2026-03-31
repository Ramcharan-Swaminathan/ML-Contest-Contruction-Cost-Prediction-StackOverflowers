import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TABULAR_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_tabular.csv")
df = pd.read_csv(TABULAR_PATH)

# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "risk_climate")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Split by country
# ---------------------------------------------------------
df_ph = df[df['country'] == 'Philippines'].copy()
df_jp = df[df['country'] == 'Japan'].copy()

# ---------------------------------------------------------
# Categorical columns to analyze
# ---------------------------------------------------------
cat_cols = [
    'seismic_hazard_zone',
    'tropical_cyclone_wind_risk',
    'tornadoes_wind_risk',
    'flood_risk_class',
    'koppen_climate_zone'
]

# ---------------------------------------------------------
# Generate one plot per category
# ---------------------------------------------------------
for col in cat_cols:

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    fig.suptitle(
        f'Impact of {col} on Construction Cost',
        fontsize=14
    )

    # Safe sorted order (drop NaNs)
    ph_order = sorted(df_ph[col].dropna().unique())
    jp_order = sorted(df_jp[col].dropna().unique())

    # Philippines
    sns.boxplot(
        x=col,
        y='construction_cost_per_m2_usd',
        data=df_ph,
        ax=axes[0],
        palette='Blues',
        order=ph_order
    )
    axes[0].set_title('Philippines')
    axes[0].tick_params(axis='x', rotation=45)

    # Japan
    sns.boxplot(
        x=col,
        y='construction_cost_per_m2_usd',
        data=df_jp,
        ax=axes[1],
        palette='Oranges',
        order=jp_order
    )
    axes[1].set_title('Japan')
    axes[1].tick_params(axis='x', rotation=45)

    # Layout + save
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])

    output_path = os.path.join(
        PLOT_DIR,
        f"{col}_ph_vs_jp.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: {output_path}")
