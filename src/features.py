"""
features.py - Matchup-Based Differential Feature Engineering

Converts raw fighter stats into differential features for ML modeling.
This removes fighter identity and forces the model to learn relative advantages.

Strategy:
- Use pre-existing *Dif columns where available
- Compute missing differential features (Red - Blue)
- Fill ranking NaN with 99 (unranked sentinel), clip rank diffs to ±15
- Encode stance matchup, weight class, gender
- Blanket-fill all numeric diff NaN with 0 (neutral advantage)
- Produce two output CSVs: with-odds and no-odds variants
- Save feature schema JSON for inference-time column alignment

Usage:
    python src/features.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

# Skill-based differential features (always included)
SKILL_STATS = {
    # Performance stats (not in existing Dif columns)
    'sig_str_pct_diff': ('RedAvgSigStrPct', 'BlueAvgSigStrPct'),
    'td_pct_diff': ('RedAvgTDPct', 'BlueAvgTDPct'),

    # Win method diffs (not in existing Dif columns)
    'dec_maj_wins_diff': ('RedWinsByDecisionMajority', 'BlueWinsByDecisionMajority'),
    'dec_split_wins_diff': ('RedWinsByDecisionSplit', 'BlueWinsByDecisionSplit'),
    'dec_unan_wins_diff': ('RedWinsByDecisionUnanimous', 'BlueWinsByDecisionUnanimous'),
    'tko_doc_wins_diff': ('RedWinsByTKODoctorStoppage', 'BlueWinsByTKODoctorStoppage'),

    # Draws (not in existing Dif)
    'draws_diff': ('RedDraws', 'BlueDraws'),
}

# Odds-based differential features (separated to allow leakage-free modeling)
ODDS_STATS = {
    'odds_diff': ('RedOdds', 'BlueOdds'),
    'ev_diff': ('RedExpectedValue', 'BlueExpectedValue'),
    'dec_odds_diff': ('RedDecOdds', 'BlueDecOdds'),
    'sub_odds_diff': ('RSubOdds', 'BSubOdds'),
    'ko_odds_diff': ('RKOOdds', 'BKOOdds'),
}

# Combined dict for computation (both are always computed, separated at output)
PAIRED_STATS = {**SKILL_STATS, **ODDS_STATS}

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

# V2: Rate-based replacements for 3 black-box pre-existing diffs.
# These are recomputed from the per-minute rate columns instead of
# the unknown-source pre-existing columns in the raw dataset.
RATE_BASED_DIFFS = {
    'SigStrDif':    ('RedAvgSigStrLanded', 'BlueAvgSigStrLanded'),
    'AvgSubAttDif': ('RedAvgSubAtt', 'BlueAvgSubAtt'),
    'AvgTDDif':     ('RedAvgTDLanded', 'BlueAvgTDLanded'),
}

# V2: Existing diffs minus the 3 black-box columns (they get recomputed)
EXISTING_DIFF_COLS_V2 = [
    c for c in EXISTING_DIFF_COLS if c not in RATE_BASED_DIFFS
]

# Contextual numeric features to keep
CONTEXTUAL_NUMERIC = ['TitleBout', 'NumberOfRounds', 'EmptyArena']

# Sentinel value for unranked fighters
UNRANKED_SENTINEL = 99

# Clip range for rank diffs (caps sentinel-inflated extremes)
RANK_DIFF_CLIP = (-15, 15)


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
    """Fill NaN with 0 for all numeric diff columns (neutral advantage).

    Covers: existing diffs, newly computed diffs, and contextual numerics.
    """
    df = df.copy()

    print("\n[FILL] Handling missing values (blanket fill -> 0 for all diffs)...")

    # Gather all diff column names that exist in the dataframe
    diff_cols = []
    diff_cols += [c for c in EXISTING_DIFF_COLS if c in df.columns]
    diff_cols += [c for c in PAIRED_STATS.keys() if c in df.columns]

    # Also fill contextual numerics (e.g. EmptyArena)
    diff_cols += [c for c in CONTEXTUAL_NUMERIC if c in df.columns]

    total_filled = 0
    for col in diff_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            df[col] = df[col].fillna(0)
            total_filled += nan_count
            print(f"   + {col}: filled {nan_count} NaN with 0")

    if total_filled == 0:
        print("   (no NaN found in diff columns)")
    else:
        print(f"   Total: {total_filled} NaN filled across {len(diff_cols)} columns")

    return df


def compute_rank_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ranking differentials with unranked sentinel fill and clipping."""
    df = df.copy()

    lo, hi = RANK_DIFF_CLIP
    print(f"\n[RANK] Computing rank differentials (NaN -> {UNRANKED_SENTINEL}, clip to [{lo}, {hi}])...")

    for diff_name, (red_col, blue_col) in RANKING_PAIRS.items():
        if red_col in df.columns and blue_col in df.columns:
            red_filled = df[red_col].fillna(UNRANKED_SENTINEL)
            blue_filled = df[blue_col].fillna(UNRANKED_SENTINEL)
            raw_diff = red_filled - blue_filled
            df[diff_name] = raw_diff.clip(lower=lo, upper=hi)

            clipped_count = ((raw_diff < lo) | (raw_diff > hi)).sum()
            non_null_pct = (df[red_col].notna().sum() / len(df)) * 100
            clip_msg = f", clipped {clipped_count}" if clipped_count > 0 else ""
            print(f"   + {diff_name} ({non_null_pct:.1f}% had Red rank{clip_msg})")
        else:
            print(f"   ! Skipping {diff_name}: columns not found")

    return df


def encode_stance_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """Create stance matchup feature and one-hot encode."""
    df = df.copy()

    print("\n[STANCE] Encoding stance matchups...")

    # Guard: check that stance columns exist
    if 'RedStance' not in df.columns or 'BlueStance' not in df.columns:
        print("   ! WARNING: RedStance/BlueStance columns not found — skipping stance encoding")
        return df

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
    """One-hot encode WeightClass; binary encode Gender.

    BetterRank is intentionally excluded — it is redundant with *_rank_diff
    features (it's just the sign of the rank differential).
    """
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

    # BetterRank — DROPPED (redundant with rank diffs)
    if 'BetterRank' in df.columns:
        print(f"   - BetterRank: SKIPPED (redundant with *_rank_diff features)")

    return df


def create_target(df: pd.DataFrame) -> pd.Series:
    """Convert Winner to binary target (Red=1, Blue=0)."""
    print("\n[TARGET] Creating binary target...")
    y = (df['Winner'] == 'Red').astype(int)
    print(f"   Red wins (y=1): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Blue wins (y=0): {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")
    return y


def select_features(df: pd.DataFrame, include_odds: bool = True) -> pd.DataFrame:
    """Select only feature columns for X.

    Args:
        include_odds: If True, include odds-derived diff columns.
                      If False, exclude them for skill-only modeling.
    """

    # Start with existing diff columns
    feature_cols = [c for c in EXISTING_DIFF_COLS if c in df.columns]

    # Add newly computed skill diff columns (always)
    feature_cols += [c for c in SKILL_STATS.keys() if c in df.columns]

    # Conditionally add odds diff columns
    if include_odds:
        feature_cols += [c for c in ODDS_STATS.keys() if c in df.columns]

    # Add rank diff columns
    feature_cols += [c for c in RANKING_PAIRS.keys() if c in df.columns]

    # Add contextual numeric
    feature_cols += [c for c in CONTEXTUAL_NUMERIC if c in df.columns]

    # Add encoded categorical columns (dummies)
    feature_cols += [c for c in df.columns if c.startswith('stance_')]
    feature_cols += [c for c in df.columns if c.startswith('wc_')]
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
    output_dir = project_root / 'data' / 'processed'
    output_with_odds = output_dir / 'features_with_odds.csv'
    output_no_odds = output_dir / 'features_no_odds.csv'
    schema_path = output_dir / 'feature_schema.json'

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

    # ---- WITH-ODDS variant ----
    print("\n[SELECT] Selecting features (WITH odds)...")
    X_with = select_features(df, include_odds=True)
    print(f"   Selected {len(X_with.columns)} features")

    # ---- NO-ODDS variant ----
    print("\n[SELECT] Selecting features (NO odds)...")
    X_no = select_features(df, include_odds=False)
    print(f"   Selected {len(X_no.columns)} features")

    # Sanity checks (run on both variants)
    for label, X in [("with_odds", X_with), ("no_odds", X_no)]:
        print(f"\n[CHECK] Sanity checks ({label})...")

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

        # Check no BetterRank leakage
        br_cols = [c for c in X.columns if c.startswith('better_rank_')]
        if br_cols:
            print(f"   ! ERROR: BetterRank dummies found: {br_cols}")
        else:
            print("   - No BetterRank dummies (removed): YES")

    # Show feature summary
    print("\n[SUMMARY] Feature breakdown (with_odds / no_odds):")
    print(f"   - Pre-existing diffs: {len([c for c in EXISTING_DIFF_COLS if c in X_with.columns])}")
    print(f"   - Skill diffs:        {len([c for c in SKILL_STATS.keys() if c in X_with.columns])}")
    print(f"   - Odds diffs:         {len([c for c in ODDS_STATS.keys() if c in X_with.columns])} (with) / 0 (no)")
    print(f"   - Rank diffs:         {len([c for c in RANKING_PAIRS.keys() if c in X_with.columns])}")
    print(f"   - Stance dummies:     {len([c for c in X_with.columns if c.startswith('stance_')])}")
    print(f"   - Weight class:       {len([c for c in X_with.columns if c.startswith('wc_')])}")
    print(f"   Total: {len(X_with.columns)} (with odds) / {len(X_no.columns)} (no odds)")

    # Save feature schema JSON (for future inference alignment)
    schema = {
        'with_odds': list(X_with.columns),
        'no_odds': list(X_no.columns),
    }
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"\n[SCHEMA] Saved feature schema to: {schema_path}")

    # Save WITH-ODDS variant
    out_with = X_with.copy()
    out_with['target'] = y
    out_with['RedFighter'] = df['RedFighter']
    out_with['BlueFighter'] = df['BlueFighter']
    out_with['Date'] = df['Date']

    print(f"\n[SAVE] Saving WITH-ODDS features to: {output_with_odds}")
    out_with.to_csv(output_with_odds, index=False)
    print(f"   Output shape: {out_with.shape[0]:,} rows x {out_with.shape[1]} columns")

    # Save NO-ODDS variant
    out_no = X_no.copy()
    out_no['target'] = y
    out_no['RedFighter'] = df['RedFighter']
    out_no['BlueFighter'] = df['BlueFighter']
    out_no['Date'] = df['Date']

    print(f"\n[SAVE] Saving NO-ODDS features to: {output_no_odds}")
    out_no.to_csv(output_no_odds, index=False)
    print(f"   Output shape: {out_no.shape[0]:,} rows x {out_no.shape[1]} columns")

    # Show sample
    print("\n[SAMPLE] First 3 rows of key differential features (with-odds):")
    sample_cols = ['HeightDif', 'ReachDif', 'odds_diff', 'wc_rank_diff', 'target']
    sample_cols = [c for c in sample_cols if c in out_with.columns]
    print(out_with[sample_cols].head(3).to_string())

    # ================================================================
    # V2: Recompute SigStrDif, AvgSubAttDif, AvgTDDif from rate columns
    # ================================================================
    print("\n" + "=" * 60)
    print("[V2] Building rate-based diff features...")

    df_v2 = df.copy()

    for diff_name, (red_col, blue_col) in RATE_BASED_DIFFS.items():
        if red_col in df_v2.columns and blue_col in df_v2.columns:
            old_vals = df_v2[diff_name].copy()
            df_v2[diff_name] = (df_v2[red_col] - df_v2[blue_col]).fillna(0)
            delta = (old_vals.fillna(0) - df_v2[diff_name]).abs()
            print(f"   + {diff_name}: recomputed from {red_col} - {blue_col}")
            print(f"     mean delta from v1: {delta.mean():.2f}, max: {delta.max():.2f}")
        else:
            print(f"   ! {diff_name}: source columns not found, kept as-is")

    # V2 feature selection (identical logic, just uses recomputed values)
    X_with_v2 = select_features(df_v2, include_odds=True)
    X_no_v2 = select_features(df_v2, include_odds=False)

    # V2 sanity check
    for label, X in [("v2_with_odds", X_with_v2), ("v2_no_odds", X_no_v2)]:
        nan_count = X.isnull().sum().sum()
        print(f"   [{label}] NaN count: {nan_count}")

    # Save V2 schema
    schema['with_odds_v2'] = list(X_with_v2.columns)
    schema['no_odds_v2'] = list(X_no_v2.columns)
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"   [SCHEMA] Updated {schema_path.name} with v2 keys")

    # Save V2 with-odds
    output_with_odds_v2 = output_dir / 'features_with_odds_v2.csv'
    out_with_v2 = X_with_v2.copy()
    out_with_v2['target'] = y
    out_with_v2['RedFighter'] = df['RedFighter']
    out_with_v2['BlueFighter'] = df['BlueFighter']
    out_with_v2['Date'] = df['Date']
    out_with_v2.to_csv(output_with_odds_v2, index=False)
    print(f"   [SAVE] {output_with_odds_v2.name}: {out_with_v2.shape}")

    # Save V2 no-odds
    output_no_odds_v2 = output_dir / 'features_no_odds_v2.csv'
    out_no_v2 = X_no_v2.copy()
    out_no_v2['target'] = y
    out_no_v2['RedFighter'] = df['RedFighter']
    out_no_v2['BlueFighter'] = df['BlueFighter']
    out_no_v2['Date'] = df['Date']
    out_no_v2.to_csv(output_no_odds_v2, index=False)
    print(f"   [SAVE] {output_no_odds_v2.name}: {out_no_v2.shape}")

    print("\n" + "=" * 60)
    print("[DONE] Feature engineering complete!")
    print(f"   V1 with-odds: {len(X_with.columns)} features")
    print(f"   V1 no-odds:   {len(X_no.columns)} features")
    print(f"   V2 with-odds: {len(X_with_v2.columns)} features")
    print(f"   V2 no-odds:   {len(X_no_v2.columns)} features")
    print(f"   Schema saved:  {schema_path.name}")
    print(f"   Ready for modeling.")
    print("=" * 60)

    return X_with, X_no, y


if __name__ == '__main__':
    main()
