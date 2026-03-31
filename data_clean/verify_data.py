import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def verify_data():
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_tabular.csv")
    JAPAN_PATH = os.path.join(SCRIPT_DIR, "train_japan.csv")
    PHILIPPINES_PATH = os.path.join(SCRIPT_DIR, "train_philippines.csv")
    PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "post_split_eda")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print("Loading data...")
    df_orig = pd.read_csv(ORIGINAL_DATA_PATH)
    df_jp = pd.read_csv(JAPAN_PATH)
    df_ph = pd.read_csv(PHILIPPINES_PATH)
    
    # ---------------------------------------------------------
    # 1. Target Encoding Check
    # ---------------------------------------------------------
    print("\n--- Target Encoding Check ---")
    target_col = 'construction_cost_per_m2_usd'
    
    # Plot Encoded Location vs Target
    for name, df in [('Japan', df_jp), ('Philippines', df_ph)]:
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x='geolocation_name', y=target_col, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.title(f'{name}: Encoded Geolocation vs Cost')
        plt.xlabel('Encoded Geolocation Name')
        plt.ylabel('Construction Cost (USD/m2)')
        plt.savefig(os.path.join(PLOT_DIR, f'target_encoding_check_{name}.png'))
        plt.close()
        
        corr = df['geolocation_name'].corr(df[target_col])
        print(f"{name} Encoded Geolocation Correlation with Target: {corr:.4f}")

    # ---------------------------------------------------------
    # 2. Imputation Check (Wind Risk)
    # ---------------------------------------------------------
    print("\n--- Imputation Check (Wind Risk) ---")
    
    # Original Philippines Wind Risk
    ph_orig = df_orig[df_orig['country'] == 'Philippines']['tropical_cyclone_wind_risk'].dropna()
    ph_clean = df_ph['tropical_cyclone_wind_risk']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # Sort order for categorical plotting consistency
    risk_order = ['Low', 'Moderate', 'High', 'Very High'] # Assuming these values
    sns.countplot(x=ph_orig, order=risk_order, palette='Blues')
    plt.title('Original Philippines Wind Risk (Non-Missing)')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=ph_clean, order=risk_order, palette='Greens')
    plt.title('Cleaned Philippines Wind Risk (Imputed)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'imputation_check_wind_risk.png'))
    plt.close()
    
    print("Original Value Counts:")
    print(ph_orig.value_counts(normalize=True))
    print("\nCleaned Value Counts:")
    print(ph_clean.value_counts(normalize=True))
    
    # ---------------------------------------------------------
    # 3. Interaction Check (Japan)
    # ---------------------------------------------------------
    print("\n--- Interaction Check (Japan: US CPI * is_capital) ---")
    
    if 'us_cpi_x_is_capital' in df_jp.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_jp, x='us_cpi_x_is_capital', y=target_col, alpha=0.5)
        plt.title('Japan: US CPI * is_capital vs Cost')
        plt.xlabel('Interaction (US CPI * is_capital)')
        plt.ylabel('Construction Cost (USD/m2)')
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_DIR, 'interaction_check_japan.png'))
        plt.close()
        
        # Also just Boxplot for Capital vs Non-Capital to see if "Capital" has higher cost generally
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_jp, x='is_capital', y=target_col)
        plt.title('Japan: Cost Distribution by Capital Status')
        plt.xticks([0, 1], ['Non-Capital', 'Capital (Tokyo)'])
        plt.savefig(os.path.join(PLOT_DIR, 'japan_capital_boxplot.png'))
        plt.close()
        
        corr_int = df_jp['us_cpi_x_is_capital'].corr(df_jp[target_col])
        print(f"Japan Interaction Correlation: {corr_int:.4f}")
    else:
        print("Interaction term not found in Japan dataset.")

    # ---------------------------------------------------------
    # 4. Correlation Matrix
    # ---------------------------------------------------------
    print("\n--- Correlation Matrix (Top 5 Positive/Negative) ---")
    for name, df in [('Japan', df_jp), ('Philippines', df_ph)]:
        numeric_df = df.select_dtypes(include=[np.number])
        corrs = numeric_df.corrwith(df[target_col]).sort_values(ascending=False)
        print(f"\n{name} Top Correlations:")
        print(corrs.head(6))
        print(corrs.tail(5))

    # ---------------------------------------------------------
    # 5. Outlier Scan
    # ---------------------------------------------------------
    print("\n--- Outlier Scan ---")
    for name, df in [('Japan', df_jp), ('Philippines', df_ph)]:
        cost = df[target_col]
        mean_val = cost.mean()
        std_val = cost.std()
        upper_limit = mean_val + 3 * std_val
        lower_limit = mean_val - 3 * std_val
        
        outliers = df[(cost > upper_limit) | (cost < lower_limit)]
        num_outliers = len(outliers)
        
        print(f"\n{name} Stats:")
        print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}")
        print(f"  >3 Std Limit: {upper_limit:.2f}")
        print(f"  Outliers found: {num_outliers} ({num_outliers/len(df)*100:.2f}%)")
        
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=cost)
        plt.title(f'{name}: Construction Cost Boxplot')
        plt.axvline(upper_limit, color='r', linestyle='--', label='Mean + 3*Std')
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, f'outlier_scan_{name}.png'))
        plt.close()

if __name__ == "__main__":
    verify_data()
