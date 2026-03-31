import pandas as pd
import numpy as np

# --- Config ---
# Student (Champion) - Score: 0.22127
FILE_STUDENT = 'submission013.csv' 
WEIGHT_STUDENT = 0.90

# Teacher (Robust Ensemble) - Score: 0.22164
FILE_TEACHER = 'submission_final.csv'
WEIGHT_TEACHER = 0.10

OUTPUT_FILE = 'submission018.csv'

def ensemble():
    print("Loading Submissions...")
    try:
        df_student = pd.read_csv(FILE_STUDENT)
        df_teacher = pd.read_csv(FILE_TEACHER)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Check alignment
    if len(df_student) != len(df_teacher):
        print("Error: Submissions vary in length!")
        return
        
    # Sort to ensure alignment
    if 'data_id' in df_student.columns:
        df_student = df_student.sort_values('data_id').reset_index(drop=True)
        df_teacher = df_teacher.sort_values('data_id').reset_index(drop=True)

        if not df_student['data_id'].equals(df_teacher['data_id']):
             print("Error: IDs do not match!")
             return

    print("Blending in Log Space...")
    # 1. Log transform
    log_pred_student = np.log1p(df_student['construction_cost_per_m2_usd'])
    log_pred_teacher = np.log1p(df_teacher['construction_cost_per_m2_usd'])
    
    # 2. Weighted Average
    final_log_pred = (WEIGHT_STUDENT * log_pred_student) + (WEIGHT_TEACHER * log_pred_teacher)
    
    # 3. Expm1
    final_pred = np.expm1(final_log_pred)
    
    # Save
    sub = pd.DataFrame()
    if 'data_id' in df_student.columns:
        sub['data_id'] = df_student['data_id']
    sub['construction_cost_per_m2_usd'] = final_pred
    
    sub.to_csv(OUTPUT_FILE, index=False)
    print(f"Safety Ensemble Saved to {OUTPUT_FILE}")
    print(f"Recipe: {WEIGHT_STUDENT} * Student + {WEIGHT_TEACHER} * Teacher")

if __name__ == "__main__":
    ensemble()
