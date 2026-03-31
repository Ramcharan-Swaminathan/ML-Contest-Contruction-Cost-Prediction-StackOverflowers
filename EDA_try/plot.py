import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TABULAR_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_tabular.csv")
df = pd.read_csv(TABULAR_PATH)
# Check if 'country' explains the two peaks
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df, 
    x='construction_cost_per_m2_usd', 
    hue='country',  # This will color the bars by country
    element="step", 
    bins=50
)
plt.title('The "Twin Peaks" Explained by Country')
#save the plot 

plt.savefig(os.path.join(PROJECT_ROOT, "twin_peaks_explained_by_country.png"))
plt.show()