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
# Filter for Japan
# ---------------------------------------------------------
df_jp = df[df['country'] == 'Philippines'].copy()

# ---------------------------------------------------------
# 1. Prepare Data (City-level statistics)
# ---------------------------------------------------------
# Median cost per city + project count
city_stats = (
    df_jp
    .groupby('geolocation_name')['construction_cost_per_m2_usd']
    .agg(['median', 'count'])
)

# Keep only cities with at least 5 projects
city_stats = city_stats[city_stats['count'] >= 5]

# Top 5 most expensive & bottom 5 cheapest cities
top_5 = city_stats.sort_values('median', ascending=False).head(5)
bottom_5 = city_stats.sort_values('median', ascending=True).head(5)

combined = pd.concat([top_5, bottom_5])

# ---------------------------------------------------------
# 2. Visualize the Gap (SAVE instead of show)
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

sns.barplot(
    x=combined.index,
    y=combined['median'],
    hue=combined.index,   # avoids seaborn palette warning
    palette='viridis',
    legend=False
)

plt.xticks(rotation=45, ha='right')
plt.title('Does City Matter? (Most Expensive vs. Cheapest Cities in Philippines)')
plt.ylabel('Median Construction Cost (USD/m²)')

# National average reference line
plt.axhline(
    df_jp['construction_cost_per_m2_usd'].mean(),
    color='red',
    linestyle='--',
    label='National Average'
)

plt.legend()

# ---------------------------------------------------------
# Save plot
# ---------------------------------------------------------
plt.savefig(
    "Philippines_city_cost_extremes.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
