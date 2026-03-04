"""
predict.py - Predict UFC Fight Outcome (Dual-Mode)

Mode A (Historical):  When the exact fighter pair exists in the
                      features dataset, uses the real fight-level row
                      with real odds → with-odds HGB model.

Mode B (Synthetic):   When the pair does NOT exist, constructs a
                      synthetic matchup from each fighter's latest
                      individual stats → no-odds HGB model.

Usage:
    python src/predict.py --red "Israel Adesanya" --blue "Sean Strickland"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_WITH_ODDS = PROJECT_ROOT / "models" / "hgb_day4a.joblib"
MODEL_NO_ODDS = PROJECT_ROOT / "models" / "hgb_no_odds.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_with_odds.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "ufc_cleaned.csv"
SCHEMA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_schema.json"

# Must match train_tree_day4a.py configuration
CLOSE_ODDS_THRESHOLD = 0.75

# Key diffs to expose for explainability charts
EXPLAINABILITY_DIFFS = {
    "ReachDif": "Reach",
    "AgeDif": "Age",
    "SigStrDif": "Sig. Strikes",
    "AvgTDDif": "Takedowns",
    "AvgSubAttDif": "Sub. Attempts",
    "HeightDif": "Height",
    "WinStreakDif": "Win Streak",
    "wc_rank_diff": "WC Rank",
    "pfp_rank_diff": "P4P Rank",
}
EXCLUDE_COLS = ["target", "RedFighter", "BlueFighter", "Date", "stance_matchup"]

# Mapping: pre-existing Dif columns → (Red column, Blue column)
EXISTING_DIFF_PAIRS = {
    "LoseStreakDif": ("RedCurrentLoseStreak", "BlueCurrentLoseStreak"),
    "WinStreakDif": ("RedCurrentWinStreak", "BlueCurrentWinStreak"),
    "LongestWinStreakDif": ("RedLongestWinStreak", "BlueLongestWinStreak"),
    "WinDif": ("RedWins", "BlueWins"),
    "LossDif": ("RedLosses", "BlueLosses"),
    "TotalRoundDif": ("RedTotalRoundsFought", "BlueTotalRoundsFought"),
    "TotalTitleBoutDif": ("RedTotalTitleBouts", "BlueTotalTitleBouts"),
    "KODif": ("RedWinsByKO", "BlueWinsByKO"),
    "SubDif": ("RedWinsBySubmission", "BlueWinsBySubmission"),
    "HeightDif": ("RedHeightCms", "BlueHeightCms"),
    "ReachDif": ("RedReachCms", "BlueReachCms"),
    "AgeDif": ("RedAge", "BlueAge"),
    "SigStrDif": ("RedAvgSigStrLanded", "BlueAvgSigStrLanded"),
    "AvgSubAttDif": ("RedAvgSubAtt", "BlueAvgSubAtt"),
    "AvgTDDif": ("RedAvgTDLanded", "BlueAvgTDLanded"),
}

# Computed skill diffs (mirrors features.py SKILL_STATS)
SKILL_STATS = {
    "sig_str_pct_diff": ("RedAvgSigStrPct", "BlueAvgSigStrPct"),
    "td_pct_diff": ("RedAvgTDPct", "BlueAvgTDPct"),
    "dec_maj_wins_diff": ("RedWinsByDecisionMajority", "BlueWinsByDecisionMajority"),
    "dec_split_wins_diff": ("RedWinsByDecisionSplit", "BlueWinsByDecisionSplit"),
    "dec_unan_wins_diff": ("RedWinsByDecisionUnanimous", "BlueWinsByDecisionUnanimous"),
    "tko_doc_wins_diff": ("RedWinsByTKODoctorStoppage", "BlueWinsByTKODoctorStoppage"),
    "draws_diff": ("RedDraws", "BlueDraws"),
}

# Ranking pairs (mirrors features.py RANKING_PAIRS)
RANKING_PAIRS = {
    "wc_rank_diff": ("RMatchWCRank", "BMatchWCRank"),
    "pfp_rank_diff": ("RPFPRank", "BPFPRank"),
    "hw_rank_diff": ("RHeavyweightRank", "BHeavyweightRank"),
    "lhw_rank_diff": ("RLightHeavyweightRank", "BLightHeavyweightRank"),
    "mw_rank_diff": ("RMiddleweightRank", "BMiddleweightRank"),
    "ww_rank_diff": ("RWelterweightRank", "BWelterweightRank"),
    "lw_rank_diff": ("RLightweightRank", "BLightweightRank"),
    "fw_rank_diff": ("RFeatherweightRank", "BFeatherweightRank"),
    "bw_rank_diff": ("RBantamweightRank", "BBantamweightRank"),
    "flw_rank_diff": ("RFlyweightRank", "BFlyweightRank"),
    "w_sw_rank_diff": ("RWStrawweightRank", "BWStrawweightRank"),
    "w_flw_rank_diff": ("RWFlyweightRank", "BWFlyweightRank"),
    "w_bw_rank_diff": ("RWBantamweightRank", "BWBantamweightRank"),
    "w_fw_rank_diff": ("RWFeatherweightRank", "BWFeatherweightRank"),
}

UNRANKED_SENTINEL = 99
RANK_DIFF_CLIP = (-15, 15)


# ============================================================================
# INTERACTION FEATURES (mirrors train_tree_day4a.build_interactions)
# ============================================================================

def build_interactions(df: pd.DataFrame, odds_col: str = "odds_diff",
                       t: float = CLOSE_ODDS_THRESHOLD) -> pd.DataFrame:
    """
    Build the same interaction features used during training.

    Mirrors train_tree_day4a.build_interactions() exactly so the feature
    vector at inference matches the feature vector at training time.
    """
    out = df.copy()

    if odds_col not in out.columns:
        # If odds are missing, fill gates and interactions with 0
        out["close_odds"] = 0
        out["red_fav_gate"] = 0
        out["red_underdog_gate"] = 0
        for col in ["reach_x_close", "age_x_close", "td_x_close",
                     "sig_x_close", "sub_x_close", "td_x_redfav",
                     "reach_x_redfav", "td_x_underdog",
                     "reach_x_underdog", "sig_x_underdog"]:
            out[col] = 0
        return out

    # Gates
    out["close_odds"] = (out[odds_col].abs() < t).astype(int)
    out["red_fav_gate"] = (out[odds_col] > 0).astype(int)
    out["red_underdog_gate"] = (out[odds_col] < 0).astype(int)

    # Close-odds interactions
    close_interactions = {
        "ReachDif": "reach_x_close",
        "AgeDif": "age_x_close",
        "AvgTDDif": "td_x_close",
        "SigStrDif": "sig_x_close",
        "AvgSubAttDif": "sub_x_close",
    }
    for base_col, new_col in close_interactions.items():
        out[new_col] = out.get(base_col, 0) * out["close_odds"]

    # Favorite interactions
    fav_interactions = {"AvgTDDif": "td_x_redfav", "ReachDif": "reach_x_redfav"}
    for base_col, new_col in fav_interactions.items():
        out[new_col] = out.get(base_col, 0) * out["red_fav_gate"]

    # Underdog interactions
    underdog_interactions = {
        "AvgTDDif": "td_x_underdog",
        "ReachDif": "reach_x_underdog",
        "SigStrDif": "sig_x_underdog",
    }
    for base_col, new_col in underdog_interactions.items():
        out[new_col] = out.get(base_col, 0) * out["red_underdog_gate"]

    return out


# ============================================================================
# HISTORICAL LOOKUP (Mode A)
# ============================================================================

def find_fight_row(df: pd.DataFrame, red: str, blue: str) -> pd.Series | None:
    """
    Find the most recent row matching the given Red/Blue fighter pair.

    Returns the matched row as a Series, or None if not found.
    """
    red_lower = red.strip().lower()
    blue_lower = blue.strip().lower()

    mask = (
        (df["RedFighter"].str.lower().str.strip() == red_lower)
        & (df["BlueFighter"].str.lower().str.strip() == blue_lower)
    )

    matches = df[mask]

    if matches.empty:
        # Try the reverse (maybe Red/Blue were swapped in the dataset)
        mask_rev = (
            (df["RedFighter"].str.lower().str.strip() == blue_lower)
            & (df["BlueFighter"].str.lower().str.strip() == red_lower)
        )
        matches = df[mask_rev]

    if matches.empty:
        return None

    # Pick the most recent fight
    if "Date" in matches.columns:
        matches = matches.copy()
        matches["_date"] = pd.to_datetime(matches["Date"], errors="coerce")
        latest = matches.sort_values("_date", ascending=False).iloc[0]
        latest = latest.drop("_date")
    else:
        latest = matches.iloc[-1]

    return latest


def enforce_schema_lock(row_df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    """
    Enforce strict column alignment between the feature row and the model.
    """
    row_set = set(row_df.columns)
    model_set = set(model_cols)

    missing = sorted(model_set - row_set)
    extra = sorted(row_set - model_set)

    if missing or extra:
        msg_parts = ["Schema lock violation -- feature drift detected!"]
        if missing:
            msg_parts.append(f"Missing ({len(missing)}): {missing[:10]}")
        if extra:
            msg_parts.append(f"Extra ({len(extra)}): {extra[:10]}")
        msg_parts.append(f"Model expects {len(model_cols)}, got {len(row_df.columns)}")
        raise ValueError(" | ".join(msg_parts))

    return row_df[model_cols]


# ============================================================================
# SYNTHETIC MATCHUP (Mode B)
# ============================================================================

def get_fighter_profile(df: pd.DataFrame, name: str) -> dict:
    """
    Find a fighter's most recent appearance and extract their stats.

    Searches both RedFighter and BlueFighter columns, returns a
    normalized dict of stats keyed by generic column names (without
    Red/Blue prefix).
    """
    name_lower = name.strip().lower()

    # Check Red side
    red_mask = df["RedFighter"].str.lower().str.strip() == name_lower
    # Check Blue side
    blue_mask = df["BlueFighter"].str.lower().str.strip() == name_lower

    red_matches = df[red_mask]
    blue_matches = df[blue_mask]

    if red_matches.empty and blue_matches.empty:
        raise ValueError(
            f"Fighter '{name}' not found in the dataset. "
            f"Name must match exactly (case-insensitive)."
        )

    # Combine all appearances and pick the most recent
    all_appearances = []

    if not red_matches.empty:
        red_matches = red_matches.copy()
        red_matches["_date"] = pd.to_datetime(red_matches["Date"], errors="coerce")
        red_matches["_side"] = "Red"
        all_appearances.append(red_matches)

    if not blue_matches.empty:
        blue_matches = blue_matches.copy()
        blue_matches["_date"] = pd.to_datetime(blue_matches["Date"], errors="coerce")
        blue_matches["_side"] = "Blue"
        all_appearances.append(blue_matches)

    combined = pd.concat(all_appearances).sort_values("_date", ascending=False)
    latest = combined.iloc[0]
    side = latest["_side"]

    # Extract per-fighter stats based on which side they appeared on
    profile = {"name": name, "side": side, "date": str(latest.get("Date", "unknown"))}

    # Map Red/Blue columns to generic names
    # Per-fighter stat columns (Red* or Blue* prefix)
    red_stat_cols = {
        "CurrentLoseStreak": "RedCurrentLoseStreak",
        "CurrentWinStreak": "RedCurrentWinStreak",
        "Draws": "RedDraws",
        "AvgSigStrLanded": "RedAvgSigStrLanded",
        "AvgSigStrPct": "RedAvgSigStrPct",
        "AvgSubAtt": "RedAvgSubAtt",
        "AvgTDLanded": "RedAvgTDLanded",
        "AvgTDPct": "RedAvgTDPct",
        "LongestWinStreak": "RedLongestWinStreak",
        "Losses": "RedLosses",
        "TotalRoundsFought": "RedTotalRoundsFought",
        "TotalTitleBouts": "RedTotalTitleBouts",
        "WinsByDecisionMajority": "RedWinsByDecisionMajority",
        "WinsByDecisionSplit": "RedWinsByDecisionSplit",
        "WinsByDecisionUnanimous": "RedWinsByDecisionUnanimous",
        "WinsByKO": "RedWinsByKO",
        "WinsBySubmission": "RedWinsBySubmission",
        "WinsByTKODoctorStoppage": "RedWinsByTKODoctorStoppage",
        "Wins": "RedWins",
        "Stance": "RedStance",
        "HeightCms": "RedHeightCms",
        "ReachCms": "RedReachCms",
        "WeightLbs": "RedWeightLbs",
        "Age": "RedAge",
    }

    blue_stat_cols = {
        "CurrentLoseStreak": "BlueCurrentLoseStreak",
        "CurrentWinStreak": "BlueCurrentWinStreak",
        "Draws": "BlueDraws",
        "AvgSigStrLanded": "BlueAvgSigStrLanded",
        "AvgSigStrPct": "BlueAvgSigStrPct",
        "AvgSubAtt": "BlueAvgSubAtt",
        "AvgTDLanded": "BlueAvgTDLanded",
        "AvgTDPct": "BlueAvgTDPct",
        "LongestWinStreak": "BlueLongestWinStreak",
        "Losses": "BlueLosses",
        "TotalRoundsFought": "BlueTotalRoundsFought",
        "TotalTitleBouts": "BlueTotalTitleBouts",
        "WinsByDecisionMajority": "BlueWinsByDecisionMajority",
        "WinsByDecisionSplit": "BlueWinsByDecisionSplit",
        "WinsByDecisionUnanimous": "BlueWinsByDecisionUnanimous",
        "WinsByKO": "BlueWinsByKO",
        "WinsBySubmission": "BlueWinsBySubmission",
        "WinsByTKODoctorStoppage": "BlueWinsByTKODoctorStoppage",
        "Wins": "BlueWins",
        "Stance": "BlueStance",
        "HeightCms": "BlueHeightCms",
        "ReachCms": "BlueReachCms",
        "WeightLbs": "BlueWeightLbs",
        "Age": "BlueAge",
    }

    # Ranking columns use R/B prefixes (not Red/Blue)
    red_rank_cols = {
        "MatchWCRank": "RMatchWCRank",
        "PFPRank": "RPFPRank",
        "HeavyweightRank": "RHeavyweightRank",
        "LightHeavyweightRank": "RLightHeavyweightRank",
        "MiddleweightRank": "RMiddleweightRank",
        "WelterweightRank": "RWelterweightRank",
        "LightweightRank": "RLightweightRank",
        "FeatherweightRank": "RFeatherweightRank",
        "BantamweightRank": "RBantamweightRank",
        "FlyweightRank": "RFlyweightRank",
        "WStrawweightRank": "RWStrawweightRank",
        "WFlyweightRank": "RWFlyweightRank",
        "WBantamweightRank": "RWBantamweightRank",
        "WFeatherweightRank": "RWFeatherweightRank",
    }

    blue_rank_cols = {
        "MatchWCRank": "BMatchWCRank",
        "PFPRank": "BPFPRank",
        "HeavyweightRank": "BHeavyweightRank",
        "LightHeavyweightRank": "BLightHeavyweightRank",
        "MiddleweightRank": "BMiddleweightRank",
        "WelterweightRank": "BWelterweightRank",
        "LightweightRank": "BLightweightRank",
        "FeatherweightRank": "BFeatherweightRank",
        "BantamweightRank": "BBantamweightRank",
        "FlyweightRank": "BFlyweightRank",
        "WStrawweightRank": "BWStrawweightRank",
        "WFlyweightRank": "BWFlyweightRank",
        "WBantamweightRank": "BWBantamweightRank",
        "WFeatherweightRank": "BWFeatherweightRank",
    }

    stat_map = red_stat_cols if side == "Red" else blue_stat_cols
    rank_map = red_rank_cols if side == "Red" else blue_rank_cols

    for generic, actual in stat_map.items():
        profile[generic] = latest.get(actual, np.nan)

    for generic, actual in rank_map.items():
        profile[generic] = latest.get(actual, np.nan)

    # Also grab Weight class and Gender from the fight context
    profile["WeightClass"] = latest.get("WeightClass", None)
    profile["Gender"] = latest.get("Gender", None)

    return profile


def build_synthetic_row(
    red_profile: dict,
    blue_profile: dict,
    weight_class: str | None = None,
    title_bout: bool = False,
    num_rounds: int = 3,
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame with all differential features
    from two fighter profiles.

    Produces the no-odds feature schema (no odds, no interaction features).
    """
    row = {}

    # ---- Pre-existing Dif columns ----
    # NOTE: SigStrDif, AvgSubAttDif, AvgTDDif are pre-computed in the raw
    # dataset (ufc-master.csv) from UNKNOWN source columns that are NOT
    # present in the CSV.  The training pipeline uses these as-is.
    # Here we recompute them from the available per-minute rate columns
    # (AvgSigStrLanded, AvgSubAtt, AvgTDLanded).  This produces values
    # on a DIFFERENT SCALE than training — the OOD guardrail below
    # mitigates extreme values until a retrain aligns the scales.
    stat_to_generic = {
        "LoseStreakDif": ("CurrentLoseStreak", "CurrentLoseStreak"),
        "WinStreakDif": ("CurrentWinStreak", "CurrentWinStreak"),
        "LongestWinStreakDif": ("LongestWinStreak", "LongestWinStreak"),
        "WinDif": ("Wins", "Wins"),
        "LossDif": ("Losses", "Losses"),
        "TotalRoundDif": ("TotalRoundsFought", "TotalRoundsFought"),
        "TotalTitleBoutDif": ("TotalTitleBouts", "TotalTitleBouts"),
        "KODif": ("WinsByKO", "WinsByKO"),
        "SubDif": ("WinsBySubmission", "WinsBySubmission"),
        "HeightDif": ("HeightCms", "HeightCms"),
        "ReachDif": ("ReachCms", "ReachCms"),
        "AgeDif": ("Age", "Age"),
        "SigStrDif": ("AvgSigStrLanded", "AvgSigStrLanded"),
        "AvgSubAttDif": ("AvgSubAtt", "AvgSubAtt"),
        "AvgTDDif": ("AvgTDLanded", "AvgTDLanded"),
    }

    for diff_name, (red_key, blue_key) in stat_to_generic.items():
        red_val = _safe_float(red_profile.get(red_key, 0))
        blue_val = _safe_float(blue_profile.get(blue_key, 0))
        row[diff_name] = red_val - blue_val

    # ---- Computed skill diffs ----
    skill_generic = {
        "sig_str_pct_diff": ("AvgSigStrPct", "AvgSigStrPct"),
        "td_pct_diff": ("AvgTDPct", "AvgTDPct"),
        "dec_maj_wins_diff": ("WinsByDecisionMajority", "WinsByDecisionMajority"),
        "dec_split_wins_diff": ("WinsByDecisionSplit", "WinsByDecisionSplit"),
        "dec_unan_wins_diff": ("WinsByDecisionUnanimous", "WinsByDecisionUnanimous"),
        "tko_doc_wins_diff": ("WinsByTKODoctorStoppage", "WinsByTKODoctorStoppage"),
        "draws_diff": ("Draws", "Draws"),
    }

    for diff_name, (red_key, blue_key) in skill_generic.items():
        red_val = _safe_float(red_profile.get(red_key, 0))
        blue_val = _safe_float(blue_profile.get(blue_key, 0))
        row[diff_name] = red_val - blue_val

    # ---- Rank differentials ----
    rank_generic = {
        "wc_rank_diff": "MatchWCRank",
        "pfp_rank_diff": "PFPRank",
        "hw_rank_diff": "HeavyweightRank",
        "lhw_rank_diff": "LightHeavyweightRank",
        "mw_rank_diff": "MiddleweightRank",
        "ww_rank_diff": "WelterweightRank",
        "lw_rank_diff": "LightweightRank",
        "fw_rank_diff": "FeatherweightRank",
        "bw_rank_diff": "BantamweightRank",
        "flw_rank_diff": "FlyweightRank",
        "w_sw_rank_diff": "WStrawweightRank",
        "w_flw_rank_diff": "WFlyweightRank",
        "w_bw_rank_diff": "WBantamweightRank",
        "w_fw_rank_diff": "WFeatherweightRank",
    }

    lo, hi = RANK_DIFF_CLIP
    for diff_name, generic_key in rank_generic.items():
        red_val = red_profile.get(generic_key, np.nan)
        blue_val = blue_profile.get(generic_key, np.nan)
        red_filled = UNRANKED_SENTINEL if pd.isna(red_val) else float(red_val)
        blue_filled = UNRANKED_SENTINEL if pd.isna(blue_val) else float(blue_val)
        raw_diff = red_filled - blue_filled
        row[diff_name] = max(lo, min(hi, raw_diff))

    # ---- Contextual numerics ----
    row["TitleBout"] = int(title_bout)
    row["NumberOfRounds"] = num_rounds
    row["EmptyArena"] = 0  # sensible default

    # ---- Stance matchup (one-hot) ----
    red_stance = red_profile.get("Stance", "Unknown")
    blue_stance = blue_profile.get("Stance", "Unknown")
    if pd.isna(red_stance) or red_stance is None:
        red_stance = "Unknown"
    if pd.isna(blue_stance) or blue_stance is None:
        blue_stance = "Unknown"
    stance_matchup = f"{red_stance}_vs_{blue_stance}"

    # All known stance dummy columns from the training schema
    all_stance_dummies = [
        "stance_Open Stance_vs_Orthodox",
        "stance_Open Stance_vs_Southpaw",
        "stance_Orthodox_vs_Orthodox",
        "stance_Orthodox_vs_Southpaw",
        "stance_Orthodox_vs_Switch",
        "stance_Orthodox_vs_Unknown",
        "stance_Southpaw_vs_Open Stance",
        "stance_Southpaw_vs_Orthodox",
        "stance_Southpaw_vs_Southpaw",
        "stance_Southpaw_vs_Switch",
        "stance_Southpaw_vs_Unknown",
        "stance_Switch_vs_Orthodox",
        "stance_Switch_vs_Southpaw",
        "stance_Switch_vs_Switch",
        "stance_Switch_vs_Switch ",  # trailing space from training data
    ]

    for dummy in all_stance_dummies:
        # dummy format is "stance_{matchup}"
        dummy_matchup = dummy.replace("stance_", "", 1)
        row[dummy] = 1 if dummy_matchup == stance_matchup else 0

    # ---- Weight class (one-hot) ----
    # Determine weight class: user override > auto-detect from fighters
    if weight_class is None:
        weight_class = red_profile.get("WeightClass") or blue_profile.get("WeightClass")

    all_wc_dummies = [
        "wc_Bantamweight",
        "wc_Catch Weight",
        "wc_Featherweight",
        "wc_Flyweight",
        "wc_Heavyweight",
        "wc_Light Heavyweight",
        "wc_Lightweight",
        "wc_Middleweight",
        "wc_Welterweight",
        "wc_Women's Bantamweight",
        "wc_Women's Featherweight",
        "wc_Women's Flyweight",
        "wc_Women's Strawweight",
    ]

    for dummy in all_wc_dummies:
        wc_name = dummy.replace("wc_", "", 1)
        row[dummy] = 1 if weight_class == wc_name else 0

    # ---- Gender ----
    gender = red_profile.get("Gender") or blue_profile.get("Gender") or "MALE"
    row["is_male"] = 1 if gender == "MALE" else 0

    return pd.DataFrame([row])


def _safe_float(val) -> float:
    """Convert value to float, treating NaN/None as 0."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ============================================================================
# OOD GUARDRAIL — clip extreme synthetic diffs + warn
# ============================================================================

# Fixed caps for high-risk diff features.
# Rate-based diffs (per-minute stats): tight cap.
# Count-based diffs (career totals): wider cap derived from training p1/p99.
OOD_CAPS = {
    "SigStrDif":    (-10, 10),
    "AvgTDDif":     (-10, 10),
    "AvgSubAttDif": (-10, 10),
    "SubDif":       (-10, 10),
    "KODif":        (-10, 10),
}


def check_ood_features(row_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip high-risk diff features to fixed caps and print warnings.

    This prevents extreme synthetic values from producing
    overconfident predictions.  Rank diffs are excluded (already
    clipped in build_synthetic_row).
    """
    row_df = row_df.copy()
    for col, (lo, hi) in OOD_CAPS.items():
        if col not in row_df.columns:
            continue
        val = row_df[col].iloc[0]
        if val < lo or val > hi:
            clipped = max(lo, min(hi, val))
            print(
                f"[OOD WARNING] {col} = {val:.4f} outside [{lo}, {hi}] "
                f"-> clipped to {clipped:.4f}"
            )
            row_df[col] = clipped
    return row_df


# ============================================================================
# ARTIFACT LOADING (cached at module level)
# ============================================================================

_cached_artifacts: dict | None = None


def _load_artifacts() -> dict:
    """
    Load models, feature data, and schema once per process.

    Returns a dict with keys:
        model_with_odds, model_cols_with_odds,
        model_no_odds, model_cols_no_odds,
        features_df, cleaned_df, schema
    """
    global _cached_artifacts
    if _cached_artifacts is not None:
        return _cached_artifacts

    # Validate required files
    for path, label, hint in [
        (MODEL_WITH_ODDS, "With-odds model", "python src/train_tree_day4a.py"),
        (MODEL_NO_ODDS, "No-odds model", "python src/train_no_odds.py"),
        (FEATURES_PATH, "Feature data", "python src/features.py"),
        (CLEANED_PATH, "Cleaned data", "python src/clean.py"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path} — Run: {hint}")

    bundle_odds = load(MODEL_WITH_ODDS)
    bundle_no_odds = load(MODEL_NO_ODDS)
    features_df = pd.read_csv(FEATURES_PATH)
    cleaned_df = pd.read_csv(CLEANED_PATH)

    _cached_artifacts = {
        "model_with_odds": bundle_odds["model"],
        "model_cols_with_odds": bundle_odds["feature_cols"],
        "model_no_odds": bundle_no_odds["model"],
        "model_cols_no_odds": bundle_no_odds["feature_cols"],
        "features_df": features_df,
        "cleaned_df": cleaned_df,
    }
    return _cached_artifacts


# ============================================================================
# PUBLIC API
# ============================================================================

def get_all_fighter_names() -> list[str]:
    """Return a sorted list of unique fighter names from the cleaned dataset."""
    arts = _load_artifacts()
    df = arts["cleaned_df"]
    fighters = sorted(
        set(df["RedFighter"].dropna().unique()) | set(df["BlueFighter"].dropna().unique())
    )
    return fighters


def predict_matchup(
    red: str,
    blue: str,
    weight_class: str | None = None,
    title_bout: bool = False,
    num_rounds: int = 3,
) -> dict:
    """
    Predict the outcome for *red* vs *blue*.

    Mode A: If the exact matchup exists in features_with_odds.csv,
            uses the historical row with the with-odds model.
    Mode B: If not, constructs a synthetic matchup from each fighter's
            latest stats and uses the no-odds model.

    Returns a dict with:
        red, blue, proba_red, predicted_winner, winner_name,
        mode ("historical" or "synthetic"), fight_date
    """
    arts = _load_artifacts()

    # ---- Try Mode A: Historical lookup ----
    row = find_fight_row(arts["features_df"], red, blue)

    if row is not None:
        return _predict_historical(arts, row, red, blue)
    else:
        return _predict_synthetic(arts, red, blue, weight_class, title_bout, num_rounds)

def _extract_diffs(row_df: pd.DataFrame) -> dict:
    """Extract key differential values from the feature row for charting."""
    diffs = {}
    for col, label in EXPLAINABILITY_DIFFS.items():
        if col in row_df.columns:
            val = row_df[col].iloc[0]
            if pd.notna(val):
                diffs[label] = float(val)
    return diffs


def _predict_historical(arts: dict, row: pd.Series, red: str, blue: str) -> dict:
    """Mode A: Predict using a historical fight row with the with-odds model."""
    model = arts["model_with_odds"]
    model_cols = arts["model_cols_with_odds"]
    fight_date = row.get("Date", "unknown")

    row_df = pd.DataFrame([row])

    # Drop non-feature columns
    drop_cols = [c for c in EXCLUDE_COLS if c in row_df.columns]
    row_df = row_df.drop(columns=drop_cols, errors="ignore")
    row_df = row_df.drop(
        columns=["RedFighter", "BlueFighter", "Date", "Winner"],
        errors="ignore",
    )

    # Build interaction features
    row_df = build_interactions(row_df)

    # Convert booleans to int
    bool_cols = row_df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        row_df[col] = row_df[col].astype(int)

    row_df = row_df.fillna(0)

    X = enforce_schema_lock(row_df, model_cols)

    # Extract explainability diffs before prediction
    diffs = _extract_diffs(row_df)

    # Predict
    proba = model.predict_proba(X)[0][1]
    predicted_winner = "Red" if proba >= 0.5 else "Blue"
    winner_name = red if predicted_winner == "Red" else blue

    return {
        "red": red,
        "blue": blue,
        "proba_red": float(proba),
        "predicted_winner": predicted_winner,
        "winner_name": winner_name,
        "mode": "historical",
        "fight_date": str(fight_date),
        "diffs": diffs,
    }


def _predict_synthetic(
    arts: dict, red: str, blue: str,
    weight_class: str | None, title_bout: bool, num_rounds: int,
) -> dict:
    """Mode B: Predict using a synthetic matchup with the no-odds model."""
    model = arts["model_no_odds"]
    model_cols = arts["model_cols_no_odds"]
    cleaned_df = arts["cleaned_df"]

    # Build fighter profiles from cleaned data
    red_profile = get_fighter_profile(cleaned_df, red)
    blue_profile = get_fighter_profile(cleaned_df, blue)

    # Build synthetic feature row
    row_df = build_synthetic_row(
        red_profile, blue_profile,
        weight_class=weight_class,
        title_bout=title_bout,
        num_rounds=num_rounds,
    )

    # Convert booleans to int
    bool_cols = row_df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        row_df[col] = row_df[col].astype(int)

    row_df = row_df.fillna(0)

    # OOD guardrail: clip extreme synthetic diffs before prediction
    row_df = check_ood_features(row_df)

    X = enforce_schema_lock(row_df, model_cols)

    # Extract explainability diffs before prediction
    diffs = _extract_diffs(row_df)

    # Predict
    proba = model.predict_proba(X)[0][1]
    predicted_winner = "Red" if proba >= 0.5 else "Blue"
    winner_name = red if predicted_winner == "Red" else blue

    return {
        "red": red,
        "blue": blue,
        "proba_red": float(proba),
        "predicted_winner": predicted_winner,
        "winner_name": winner_name,
        "mode": "synthetic",
        "fight_date": "upcoming",
        "red_data_from": red_profile["date"],
        "blue_data_from": blue_profile["date"],
        "diffs": diffs,
    }


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict UFC fight outcome using the trained HGB model."
    )
    parser.add_argument("--red", required=True, help="Red corner fighter name")
    parser.add_argument("--blue", required=True, help="Blue corner fighter name")
    parser.add_argument("--weight-class", default=None, help="Weight class override")
    parser.add_argument("--title-bout", action="store_true", help="Title bout flag")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds (3 or 5)")
    args = parser.parse_args()

    print("=" * 50)
    print("UFC Fight Predictor - Inference")
    print("=" * 50)

    try:
        arts = _load_artifacts()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\n[LOAD] With-odds model: {MODEL_WITH_ODDS.name}")
    print(f"[LOAD] No-odds model:  {MODEL_NO_ODDS.name}")
    print(f"[LOAD] Features:       {FEATURES_PATH.name}")
    print(f"[LOAD] Cleaned data:   {CLEANED_PATH.name}")

    print(f"\n[FIND] Looking up: {args.red} vs {args.blue}")

    try:
        result = predict_matchup(
            args.red, args.blue,
            weight_class=args.weight_class,
            title_bout=args.title_bout,
            num_rounds=args.rounds,
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"       Mode: {result['mode'].upper()}")
    if result["mode"] == "historical":
        print(f"       Based on fight from: {result['fight_date']}")
    else:
        print(f"       Red data from: {result.get('red_data_from', '?')}")
        print(f"       Blue data from: {result.get('blue_data_from', '?')}")

    print("\n" + "-" * 50)
    print("PREDICTION")
    print("-" * 50)
    print(f"  Red corner:      {result['red']}")
    print(f"  Blue corner:     {result['blue']}")
    print(f"  P(Red wins):     {result['proba_red']:.1%}")
    print(f"  Predicted winner: {result['winner_name']} ({result['predicted_winner']} corner)")
    print("-" * 50)

    return result


if __name__ == "__main__":
    main()
