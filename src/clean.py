"""
clean.py - Leakage-Safe Data Cleaning for UFC Fight Predictor

This script removes post-fight (leakage) columns from the raw dataset
while preserving the Winner column as the target variable.

Forbidden columns (post-fight information):
- Finish: How the fight ended (KO, SUB, DEC)
- FinishDetails: Specific finish details (e.g., "Rear Naked Choke")
- FinishRound: Round the fight ended
- FinishRoundTime: Time in round when fight ended
- TotalFightTimeSecs: Total fight duration in seconds

Usage:
    python src/clean.py
"""

import pandas as pd
from pathlib import Path


# Columns that contain post-fight information (data leakage)
FORBIDDEN_COLUMNS = [
    'Finish',
    'FinishDetails', 
    'FinishRound',
    'FinishRoundTime',
    'TotalFightTimeSecs',
]

# Target variable (kept for modeling, but not used as a feature)
TARGET_COLUMN = 'Winner'


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    raw_path = project_root / 'data' / 'raw' / 'ufc-master.csv'
    processed_path = project_root / 'data' / 'processed' / 'ufc_cleaned.csv'
    
    print("=" * 60)
    print("UFC Fight Predictor - Leakage-Safe Data Cleaning")
    print("=" * 60)
    
    # Load raw data
    print(f"\n[LOAD] Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"   Raw dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Verify forbidden columns exist
    print(f"\n[CHECK] Checking for forbidden (leakage) columns...")
    found_forbidden = [col for col in FORBIDDEN_COLUMNS if col in df.columns]
    missing_forbidden = [col for col in FORBIDDEN_COLUMNS if col not in df.columns]
    
    if missing_forbidden:
        print(f"   WARNING: These forbidden columns were not found: {missing_forbidden}")
    
    print(f"   Found {len(found_forbidden)} forbidden columns to remove:")
    for col in found_forbidden:
        non_null = df[col].notna().sum()
        print(f"      - {col} ({non_null:,} non-null values)")
    
    # Verify target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset!")
    
    print(f"\n[TARGET] Target variable: '{TARGET_COLUMN}'")
    print(f"   Distribution:")
    winner_counts = df[TARGET_COLUMN].value_counts()
    for winner, count in winner_counts.items():
        pct = count / len(df) * 100
        print(f"      - {winner}: {count:,} ({pct:.1f}%)")
    
    # Drop forbidden columns
    print(f"\n[CLEAN] Dropping forbidden columns...")
    df_clean = df.drop(columns=found_forbidden, errors='ignore')
    
    columns_removed = df.shape[1] - df_clean.shape[1]
    print(f"   Removed {columns_removed} columns")
    print(f"   Cleaned dataset shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
    
    # Save cleaned data
    print(f"\n[SAVE] Saving cleaned data to: {processed_path}")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    
    # Final verification
    print(f"\n[OK] Verification:")
    print(f"   - Target column '{TARGET_COLUMN}' preserved: {TARGET_COLUMN in df_clean.columns}")
    
    leakage_check = [col for col in FORBIDDEN_COLUMNS if col in df_clean.columns]
    if leakage_check:
        print(f"   ERROR: Leakage columns still present: {leakage_check}")
    else:
        print(f"   - All leakage columns removed: YES")
    
    print(f"\n" + "=" * 60)
    print("[DONE] Data cleaning complete! Dataset is leakage-safe.")
    print("=" * 60)
    
    return df_clean


if __name__ == '__main__':
    main()
