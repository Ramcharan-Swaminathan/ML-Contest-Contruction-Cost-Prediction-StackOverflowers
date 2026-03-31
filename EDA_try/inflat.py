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
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "inflation_effects")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Filter Japan
# ---------------------------------------------------------
df_jp = df[df['country'] == 'Japan'].copy()

# ---------------------------------------------------------
# Helper: group top cities vs others
# ---------------------------------------------------------
def categorize_loc(city, top_cities):
    return city if city in top_cities else 'Rural/Other'

# ---------------------------------------------------------
# Select top 3 cities
# ---------------------------------------------------------
top_jp = (
    df_jp['geolocation_name']
    .value_counts()
    .head(3)
    .index
    .tolist()
)

df_jp['location_group'] = df_jp['geolocation_name'].apply(
    lambda x: categorize_loc(x, top_jp)
)

# ---------------------------------------------------------
# Plot: Inflation vs Cost
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

sns.lineplot(
    data=df_jp,
    x='us_cpi',
    y='construction_cost_per_m2_usd',
    hue='location_group',
    errorbar=None   # replaces deprecated ci=None
)

plt.title('Japan: Does Inflation Hit Cities Harder?')
plt.xlabel('US CPI (Inflation Proxy)')
plt.ylabel('Construction Cost (USD/m²)')
plt.legend(title='Location Group')

# ---------------------------------------------------------
# Save plot
# ---------------------------------------------------------
output_path = os.path.join(
    PLOT_DIR,
    "japan_inflation_city_vs_rural.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"Saved: {output_path}")
