import os
import json
import pandas as pd
from ydata_profiling import ProfileReport


# =========================
# PATH CONFIGURATION
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TABULAR_PATH = os.path.join(PROJECT_ROOT, "data_clean", "train_philippines.csv")
OUTPUT_HTML = os.path.join(SCRIPT_DIR, "EDA_Report_philippines.html")

TARGET_COL = "construction_cost_per_m2_usd"



# =========================
# MAIN PIPELINE
# =========================

def main():
    print("Loading dataset...")

    if not os.path.exists(TABULAR_PATH):
        raise FileNotFoundError(f"Dataset not found at {TABULAR_PATH}")

    df = pd.read_csv(TABULAR_PATH)
    print(f"Dataset shape: {df.shape}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    # Ensure target is numeric
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # -------------------------
    # Generate EDA Report
    # -------------------------
    print("Generating YData-Profiling report...")

    profile = ProfileReport(
        df,
        title="Construction Cost Regression EDA (Tabular Only)",
        explorative=True,
        variables={
            "target": TARGET_COL
        }
    )

    profile.to_file(OUTPUT_HTML)
    print(f"HTML report saved to: {OUTPUT_HTML}")




if __name__ == "__main__":
    main()
