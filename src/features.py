"""
features.py - Matchup-Based Differential Feature Engineering

Converts raw fighter stats into differential features for ML modeling.
This removes fighter identity and forces the model to learn relative advantages.

Strategy:
- Use pre-existing *Dif columns where available
- Compute missing differential features (Red - Blue)
- Fill ranking NaN with 99 (unranked sentinel)
- Encode stance matchup, weight class, gender, better_rank
- Exclude fighter names from feature matrix

Usage:
    python src/features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

# Columns to compute differentials for (if not already present as *Dif)
PAIRED_STATS = {
    # Performance stats (not in existing Dif columns)
    'sig_str_pct_diff': ('RedAvgSigStrPct', 'BlueAvgSigStrPct'),
    'td_pct_diff': ('RedAvgTDPct', 'BlueAvgTDPct'),
    
    # Betting odds
    'odds_diff': ('RedOdds', 'BlueOdds'),
    'ev_diff': ('RedExpectedValue', 'BlueExpectedValue'),
    'dec_odds_diff': ('RedDecOdds', 'BlueDecOdds'),
    'sub_odds_diff': ('RSubOdds', 'BSubOdds'),
    'ko_odds_diff': ('RKOOdds', 'BKOOdds'),
    
    # Win method diffs (not in existing Dif columns)
    'dec_maj_wins_diff': ('RedWinsByDecisionMajority', 'BlueWinsByDecisionMajority'),
    'dec_split_wins_diff': ('RedWinsByDecisionSplit', 'BlueWinsByDecisionSplit'),
    'dec_unan_wins_diff': ('RedWinsByDecisionUnanimous', 'BlueWinsByDecisionUnanimous'),
    'tko_doc_wins_diff': ('RedWinsByTKODoctorStoppage', 'BlueWinsByTKODoctorStoppage'),
    
    # Draws (not in existing Dif)
    'draws_diff': ('RedDraws', 'BlueDraws'),
}

# Ranking columns (use R/B prefix, fill NaN with 99)
RANKING_PAIRS = {
    'wc_rank_diff': ('RMatchWCRank', 'BMatchWCRank'),
    'pfp_rank_diff': ('RPFPRank', 'BPFPRank'),
    # Weight class specific ranks (sparse but may help)
    'hw_rank_diff': ('RHeavyweightRank', 'BHeavyweightRank'),
    'lhw_rank_diff': ('RLightHeavyweightRank', 'BLightHeavyweightRank'),
    'mw_rank_diff': ('RMiddleweightRank', 'BMiddleweightRank'),
    'ww_rank_diff': ('RWelterweightRank', 'BWelterweightRank'),
    'lw_rank_diff': ('RLightweightRank', 'BLightweightRank'),
    'fw_rank_diff': ('RFeatherweightRank', 'BFeatherweightRank'),
    'bw_rank_diff': ('RBantamweightRank', 'BBantamweightRank'),
    'flw_rank_diff': ('RFlyweightRank', 'BFlyweightRank'),
    # Women's divisions
    'w_sw_rank_diff': ('RWStrawweightRank', 'BWStrawweightRank'),
    'w_flw_rank_diff': ('RWFlyweightRank', 'BWFlyweightRank'),
    'w_bw_rank_diff': ('RWBantamweightRank', 'BWBantamweightRank'),
    'w_fw_rank_diff': ('RWFeatherweightRank', 'BWFeatherweightRank'),
}

# Pre-existing differential columns (use as-is)
EXISTING_DIFF_COLS = [
    'LoseStreakDif', 'WinStreakDif', 'LongestWinStreakDif',
    'WinDif', 'LossDif', 'TotalRoundDif', 'TotalTitleBoutDif',
    'KODif', 'SubDif', 'HeightDif', 'ReachDif', 'AgeDif',
    'SigStrDif', 'AvgSubAttDif', 'AvgTDDif'
]

# Contextual numeric features to keep
CONTEXTUAL_NUMERIC = ['TitleBout', 'NumberOfRounds', 'EmptyArena']

# Columns to exclude from features (keep for display only)
EXCLUDE_FROM_X = ['RedFighter', 'BlueFighter', 'Date', 'Location', 'Country', 'Winner']

# Sentinel value for unranked fighters
UNRANKED_SENTINEL = 99


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def compute_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Red - Blue differential features."""
    df = df.copy()
    
    print("\n[DIFF] Computing differential features...")
    
    for diff_name, (red_col, blue_col) in PAIRED_STATS.items():
        if red_col in df.columns and blue_col in df.columns:
            df[diff_name] = df[red_col] - df[blue_col]
            print(f"   + {diff_name} = {red_col} - {blue_col}")
        else:
            print(f"   ! Skipping {diff_name}: columns not found")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle NaN values in differential and contextual columns.
    
    Strategy:
    - Performance stat diffs (sig_str_pct_diff, td_pct_diff): fill with 0 (no advantage)
    - Odds diffs: fill with 0 (no betting edge)
    - EmptyArena: fill with 0 (assume normal arena)
    """
    df = df.copy()
    
    print("\n[FILL] Handling missing values...")
    
    # Performance stat diffs - fill with 0 (neutral, no advantage)
    perf_cols = ['sig_str_pct_diff', 'td_pct_diff']
    for col in perf_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                df[col] = df[col].fillna(0)
                print(f"   + {col}: filled {nan_count} NaN with 0")
    
    # Odds diffs - fill with 0 (no betting edge)
    odds_cols = ['odds_diff', 'ev_diff', 'dec_odds_diff', 'sub_odds_diff', 'ko_odds_diff']
    for col in odds_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                df[col] = df[col].fillna(0)
                print(f"   + {col}: filled {nan_count} NaN with 0")
    
    # EmptyArena - fill with 0 (assume normal arena, pre-COVID)
    if 'EmptyArena' in df.columns:
        nan_count = df['EmptyArena'].isna().sum()
        if nan_count > 0:
            df['EmptyArena'] = df['EmptyArena'].fillna(0)
            print(f"   + EmptyArena: filled {nan_count} NaN with 0")
    
    return df


def compute_rank_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ranking differentials with unranked sentinel fill."""
    df = df.copy()
    
    print(f"\n[RANK] Computing rank differentials (NaN -> {UNRANKED_SENTINEL})...")
    
    for diff_name, (red_col, blue_col) in RANKING_PAIRS.items():
        if red_col in df.columns and blue_col in df.columns:
            red_filled = df[red_col].fillna(UNRANKED_SENTINEL)
            blue_filled = df[blue_col].fillna(UNRANKED_SENTINEL)
            df[diff_name] = red_filled - blue_filled
            
            non_null_pct = (df[red_col].notna().sum() / len(df)) * 100
            print(f"   + {diff_name} ({non_null_pct:.1f}% had Red rank)")
        else:
            print(f"   ! Skipping {diff_name}: columns not found")
    
    return df


def encode_stance_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """Create stance matchup feature and one-hot encode."""
    df = df.copy()
    
    print("\n[STANCE] Encoding stance matchups...")
    
    # Fill missing stances with 'Unknown'
    red_stance = df['RedStance'].fillna('Unknown')
    blue_stance = df['BlueStance'].fillna('Unknown')
    
    # Create matchup string
    df['stance_matchup'] = red_stance + '_vs_' + blue_stance
    
    # Show distribution
    matchup_counts = df['stance_matchup'].value_counts().head(5)
    print("   Top 5 matchups:")
    for matchup, count in matchup_counts.items():
        print(f"      - {matchup}: {count}")
    
    # One-hot encode
    stance_dummies = pd.get_dummies(df['stance_matchup'], prefix='stance')
    df = pd.concat([df, stance_dummies], axis=1)
    
    print(f"   Created {len(stance_dummies.columns)} stance dummy columns")
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode WeightClass, BetterRank; binary encode Gender."""
    df = df.copy()
    
    print("\n[CAT] Encoding categorical features...")
    
    # WeightClass - one-hot
    if 'WeightClass' in df.columns:
        wc_dummies = pd.get_dummies(df['WeightClass'], prefix='wc')
        df = pd.concat([df, wc_dummies], axis=1)
        print(f"   + WeightClass: {len(wc_dummies.columns)} classes")
    
    # Gender - binary (MALE=1, FEMALE=0)
    if 'Gender' in df.columns:
        df['is_male'] = (df['Gender'] == 'MALE').astype(int)
        print(f"   + Gender: binary encoded as is_male")
    
    # BetterRank - one-hot
    if 'BetterRank' in df.columns:
        br_dummies = pd.get_dummies(df['BetterRank'], prefix='better_rank')
        df = pd.concat([df, br_dummies], axis=1)
        print(f"   + BetterRank: {len(br_dummies.columns)} categories")
    
    return df


def create_target(df: pd.DataFrame) -> pd.Series:
    """Convert Winner to binary target (Red=1, Blue=0)."""
    print("\n[TARGET] Creating binary target...")
    y = (df['Winner'] == 'Red').astype(int)
    print(f"   Red wins (y=1): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Blue wins (y=0): {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")
    return y


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only feature columns for X."""
    
    # Start with existing diff columns
    feature_cols = [c for c in EXISTING_DIFF_COLS if c in df.columns]
    
    # Add newly computed diff columns
    feature_cols += [c for c in PAIRED_STATS.keys() if c in df.columns]
    
    # Add rank diff columns
    feature_cols += [c for c in RANKING_PAIRS.keys() if c in df.columns]
    
    # Add contextual numeric
    feature_cols += [c for c in CONTEXTUAL_NUMERIC if c in df.columns]
    
    # Add encoded categorical columns (dummies)
    feature_cols += [c for c in df.columns if c.startswith('stance_')]
    feature_cols += [c for c in df.columns if c.startswith('wc_')]
    feature_cols += [c for c in df.columns if c.startswith('better_rank_')]
    if 'is_male' in df.columns:
        feature_cols.append('is_male')
    
    # Remove duplicates while preserving order
    feature_cols = list(dict.fromkeys(feature_cols))
    
    return df[feature_cols]


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'data' / 'processed' / 'ufc_cleaned.csv'
    output_path = project_root / 'data' / 'processed' / 'ufc_features.csv'
    
    print("=" * 60)
    print("UFC Fight Predictor - Differential Feature Engineering")
    print("=" * 60)
    
    # Load cleaned data
    print(f"\n[LOAD] Loading cleaned data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Input shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Feature engineering pipeline
    df = compute_differential_features(df)
    df = handle_missing_values(df)
    df = compute_rank_differentials(df)
    df = encode_stance_matchup(df)
    df = encode_categorical_features(df)
    
    # Create target
    y = create_target(df)
    
    # Select feature columns
    print("\n[SELECT] Selecting feature columns...")
    X = select_features(df)
    print(f"   Selected {len(X.columns)} features")
    
    # Sanity checks
    print("\n[CHECK] Sanity checks...")
    
    # Check for NaN
    nan_counts = X.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"   ! WARNING: {len(cols_with_nan)} columns have NaN values:")
        for col, count in cols_with_nan.items():
            print(f"      - {col}: {count} NaN")
    else:
        print("   - No NaN values: YES")
    
    # Check no fighter names in X
    forbidden_in_X = [c for c in X.columns if 'Fighter' in c or 'fighter' in c]
    if forbidden_in_X:
        print(f"   ! ERROR: Fighter names found in X: {forbidden_in_X}")
    else:
        print("   - No fighter names in X: YES")
    
    # Check no duplicate columns
    if X.columns.duplicated().any():
        print(f"   ! ERROR: Duplicate columns found")
    else:
        print("   - No duplicate columns: YES")
    
    # Show feature summary
    print("\n[SUMMARY] Feature breakdown:")
    print(f"   - Pre-existing diffs: {len([c for c in EXISTING_DIFF_COLS if c in X.columns])}")
    print(f"   - New computed diffs: {len([c for c in PAIRED_STATS.keys() if c in X.columns])}")
    print(f"   - Rank diffs: {len([c for c in RANKING_PAIRS.keys() if c in X.columns])}")
    print(f"   - Stance dummies: {len([c for c in X.columns if c.startswith('stance_')])}")
    print(f"   - Weight class dummies: {len([c for c in X.columns if c.startswith('wc_')])}")
    print(f"   - Other: {len([c for c in X.columns if not any(c.startswith(p) for p in ['stance_', 'wc_', 'better_rank_'])])}")
    
    # Combine X and y for saving
    output_df = X.copy()
    output_df['target'] = y
    
    # Also keep fighter names and date for reference (separate from features)
    output_df['RedFighter'] = df['RedFighter']
    output_df['BlueFighter'] = df['BlueFighter']
    output_df['Date'] = df['Date']
    
    # Save
    print(f"\n[SAVE] Saving to: {output_path}")
    output_df.to_csv(output_path, index=False)
    print(f"   Output shape: {output_df.shape[0]:,} rows x {output_df.shape[1]} columns")
    print(f"   (includes target + 3 reference columns)")
    
    # Show sample
    print("\n[SAMPLE] First 3 rows of key differential features:")
    sample_cols = ['HeightDif', 'ReachDif', 'odds_diff', 'wc_rank_diff', 'target']
    sample_cols = [c for c in sample_cols if c in output_df.columns]
    print(output_df[sample_cols].head(3).to_string())
    
    print("\n" + "=" * 60)
    print("[DONE] Feature engineering complete!")
    print(f"   Total features: {len(X.columns)}")
    print(f"   Ready for modeling.")
    print("=" * 60)
    
    return X, y


if __name__ == '__main__':
    main()
