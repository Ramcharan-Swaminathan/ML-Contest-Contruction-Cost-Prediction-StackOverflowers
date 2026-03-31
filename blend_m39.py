import pandas as pd
import numpy as np

# --- Config ---
PATH_M38 = 'submission038.csv' # The Champ (0.2198)
PATH_M21 = 'phase1_archive/submission021.csv' # Visual Champ (0.2205)
PATH_KESTAV = 'phase1_archive/sub_ens_kestav.csv' # Teacher (0.22046)
OUT_PATH = 'submission039.csv'

def blend():
    print("Loading Submissions...")
    s38 = pd.read_csv(PATH_M38) # Student
    s21 = pd.read_csv(PATH_M21) # Visual
    sk  = pd.read_csv(PATH_KESTAV) # Teacher
    
    # Sort
    s38 = s38.sort_values('data_id').reset_index(drop=True)
    s21 = s21.sort_values('data_id').reset_index(drop=True)
    sk = sk.sort_values('data_id').reset_index(drop=True)
    
    # Check alignment
    if not (s38['data_id'].equals(s21['data_id']) and s38['data_id'].equals(sk['data_id'])):
        raise ValueError("Data IDs do not match!")
        
    p38 = np.log1p(s38['construction_cost_per_m2_usd'])
    p21 = np.log1p(s21['construction_cost_per_m2_usd'])
    pk  = np.log1p(sk['construction_cost_per_m2_usd'])
    
    # Correlation Matrix
    corr = np.corrcoef([p38, p21, pk])
    print("\nCorrelation Matrix:")
    print(f"M38 (Student) vs M21 (Visual):  {corr[0,1]:.5f}")
    print(f"M38 (Student) vs Kestav (Teach): {corr[0,2]:.5f}")
    print(f"M21 (Visual)  vs Kestav (Teach): {corr[1,2]:.5f}")
    
    # Strategy: Blend Student (Best) with the most diverse strong model.
    # Since M38 is 0.2198 and M21/Kestav are ~0.2205, M38 should have higher weight.
    # Proposed: 60% M38, 40% Other.
    
    # Select Partner
    # We want the one least correlated with M38.
    if corr[0,1] < corr[0,2]:
        print("\nSelecting M21 (Visual) as partner (Lower Correlation).")
        partner = p21
        w_p = 0.4
        name = "M21"
    else:
        print("\nSelecting Kestav (Teacher) as partner (Lower Correlation).")
        partner = pk
        w_p = 0.4
        name = "Kestav"
        
    w_38 = 0.6
    
    print(f"Blending: {w_38} * M38 + {w_p} * {name}")
    
    log_blend = w_38 * p38 + w_p * partner
    final_pred = np.expm1(log_blend)
    
    # Output
    sub = pd.DataFrame({'data_id': s38['data_id'], 'construction_cost_per_m2_usd': final_pred})
    sub.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    blend()
