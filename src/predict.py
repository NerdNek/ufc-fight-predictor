"""
predict.py - Predict UFC Fight Outcome

Loads the trained HGB model and predicts the outcome for a given
Red/Blue fighter pair by looking up the most recent matching row
in the feature dataset.

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
MODEL_PATH = PROJECT_ROOT / "models" / "hgb_day4a.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_with_odds.csv"
SCHEMA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_schema.json"

# Must match train_tree_day4a.py configuration
CLOSE_ODDS_THRESHOLD = 0.75
EXCLUDE_COLS = ["target", "RedFighter", "BlueFighter", "Date", "stance_matchup"]


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
# LOOKUP & ALIGNMENT
# ============================================================================

def find_fight_row(df: pd.DataFrame, red: str, blue: str) -> pd.Series:
    """
    Find the most recent row matching the given Red/Blue fighter pair.

    Searches for exact match (case-insensitive) on RedFighter and
    BlueFighter columns. If duplicates exist, returns the row with the
    latest Date.

    Returns the matched row as a Series, or raises SystemExit if not found.
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
        if not matches.empty:
            print(f"[NOTE] Found match with corners swapped "
                  f"(Red={blue}, Blue={red} in dataset)")

    if matches.empty:
        print(f"[ERROR] No fight found for:")
        print(f"   Red:  {red}")
        print(f"   Blue: {blue}")
        print(f"\nTip: Names must match the dataset exactly.")
        print(f"     Try searching with: python -c \"import pandas as pd; "
              f"df=pd.read_csv('{DATA_PATH}'); "
              f"print(df[df['RedFighter'].str.contains('LastName', case=False)]"
              f"[['RedFighter','BlueFighter','Date']].to_string())\"")
        sys.exit(1)

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

    Compares set(row_df.columns) to set(model_cols). If there is any
    mismatch, prints the missing and extra columns and exits with an
    error. This prevents silent feature drift.

    If columns match, reorders row_df to the model's expected order.
    """
    row_set = set(row_df.columns)
    model_set = set(model_cols)

    missing = sorted(model_set - row_set)
    extra = sorted(row_set - model_set)

    if missing or extra:
        print("\n[FATAL] Schema lock violation -- feature drift detected!")
        if missing:
            print(f"   Missing columns ({len(missing)}):")
            for col in missing:
                print(f"      - {col}")
        if extra:
            print(f"   Extra columns ({len(extra)}):")
            for col in extra:
                print(f"      + {col}")
        print(f"\n   Model expects {len(model_cols)} features, got {len(row_df.columns)}")
        print("   Fix: re-run features.py and re-train, or update predict.py")
        sys.exit(1)

    # Reorder to match model's exact column order
    return row_df[model_cols]


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict UFC fight outcome using the trained HGB model."
    )
    parser.add_argument("--red", required=True, help="Red corner fighter name")
    parser.add_argument("--blue", required=True, help="Blue corner fighter name")
    args = parser.parse_args()

    # ---- Load artifacts ----
    print("=" * 50)
    print("UFC Fight Predictor - Inference")
    print("=" * 50)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Run: python src/train_tree_day4a.py")
        sys.exit(1)

    if not DATA_PATH.exists():
        print(f"[ERROR] Feature data not found: {DATA_PATH}")
        print("        Run: python src/features.py")
        sys.exit(1)

    if not SCHEMA_PATH.exists():
        print(f"[ERROR] Feature schema not found: {SCHEMA_PATH}")
        print("        Run: python src/features.py")
        sys.exit(1)

    print(f"\n[LOAD] Model:   {MODEL_PATH.name}")
    artifacts = load(MODEL_PATH)
    model = artifacts["model"]
    model_cols = artifacts["feature_cols"]

    print(f"[LOAD] Data:    {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH)

    print(f"[LOAD] Schema:  {SCHEMA_PATH.name}")
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    print(f"       Schema features (with_odds): {len(schema['with_odds'])}")
    print(f"       Model expects: {len(model_cols)} features")

    # Safety: verify fighter columns exist in the dataset
    for col in ["RedFighter", "BlueFighter"]:
        if col not in df.columns:
            print(f"[FATAL] Required column '{col}' not found in {DATA_PATH.name}")
            sys.exit(1)

    # ---- Find the fight ----
    print(f"\n[FIND] Looking up: {args.red} vs {args.blue}")
    row = find_fight_row(df, args.red, args.blue)

    fight_date = row.get("Date", "unknown")
    print(f"       Found fight from: {fight_date}")

    # ---- Build feature row ----
    row_df = pd.DataFrame([row])

    # Drop non-feature columns
    drop_cols = [c for c in EXCLUDE_COLS if c in row_df.columns]
    row_df = row_df.drop(columns=drop_cols, errors="ignore")

    # Drop fighter name / date columns that may remain
    row_df = row_df.drop(
        columns=["RedFighter", "BlueFighter", "Date", "Winner"],
        errors="ignore",
    )

    # Build interaction features (must match training)
    row_df = build_interactions(row_df)

    # Convert booleans to int (matches get_X_y in training)
    bool_cols = row_df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        row_df[col] = row_df[col].astype(int)

    # Fill any remaining NaN with 0
    row_df = row_df.fillna(0)

    # Schema lock: strict column alignment (fails on mismatch)
    X = enforce_schema_lock(row_df, model_cols)

    # ---- Safety checks ----
    if len(X) != 1:
        print(f"[FATAL] Expected exactly 1 row, got {len(X)}")
        sys.exit(1)

    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        nan_cols = X.columns[X.isnull().any()].tolist()
        print(f"[FATAL] {nan_count} NaN values found in feature row:")
        for col in nan_cols:
            print(f"   - {col}")
        sys.exit(1)

    print(f"\n[CHECK] Schema lock:  PASS ({len(model_cols)} features aligned)")
    print(f"[CHECK] NaN check:    PASS (0 NaN)")
    print(f"[CHECK] Row count:    PASS (1 row)")

    # ---- Predict ----
    proba = model.predict_proba(X)[0][1]
    predicted_winner = "Red" if proba >= 0.5 else "Blue"
    winner_name = args.red if predicted_winner == "Red" else args.blue

    # ---- Output ----
    print("\n" + "-" * 50)
    print("PREDICTION")
    print("-" * 50)
    print(f"  Red corner:      {args.red}")
    print(f"  Blue corner:     {args.blue}")
    print(f"  P(Red wins):     {proba:.1%}")
    print(f"  Predicted winner: {winner_name} ({predicted_winner} corner)")
    print("-" * 50)

    return {
        "red": args.red,
        "blue": args.blue,
        "proba_red": float(proba),
        "predicted_winner": predicted_winner,
        "winner_name": winner_name,
        "fight_date": str(fight_date),
    }


if __name__ == "__main__":
    main()
