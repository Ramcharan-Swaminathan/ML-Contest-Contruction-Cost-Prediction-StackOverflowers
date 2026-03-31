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
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "distance_effects")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Filter Philippines
# ---------------------------------------------------------
df_ph = df[df['country'] == 'Philippines'].copy()

# Optional: drop rows with missing distance or cost
df_ph = df_ph.dropna(
    subset=['straight_distance_to_capital_km', 'construction_cost_per_m2_usd']
)

# ---------------------------------------------------------
# Plot: Distance vs Construction Cost (LOWESS)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

sns.regplot(
    data=df_ph,
    x='straight_distance_to_capital_km',
    y='construction_cost_per_m2_usd',
    scatter_kws={'alpha': 0.3},
    line_kws={'color': 'red'},
    lowess=True
)

plt.title('Philippines: The "Distance Decay" of Construction Cost')
plt.xlabel('Straight Distance to Capital (km)')
plt.ylabel('Construction Cost (USD/m²)')
plt.grid(True, alpha=0.3)

# ---------------------------------------------------------
# Save plot
# ---------------------------------------------------------
output_path = os.path.join(
    PLOT_DIR,
    "philippines_distance_decay_cost.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"Saved: {output_path}")
