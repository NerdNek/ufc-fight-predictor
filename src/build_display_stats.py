"""
build_display_stats.py - Compute display normalization stats for matchup chart.

Reads v1 with-odds and v2 no-odds feature files, computes mean/std/p05/p95
on the TRAIN split (oldest 80% by Date) for key diff features.

Output: reports/feature_display_stats.json

Usage:
    python src/build_display_stats.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

SOURCES = {
    "v1_with_odds": PROJECT_ROOT / "data" / "processed" / "features_with_odds.csv",
    "v2_no_odds": PROJECT_ROOT / "data" / "processed" / "features_no_odds_v2.csv",
}

OUTPUT_PATH = PROJECT_ROOT / "reports" / "feature_display_stats.json"

TRAIN_RATIO = 0.80

DISPLAY_FEATURES = [
    "ReachDif", "AgeDif", "SigStrDif", "AvgTDDif", "AvgSubAttDif",
    "HeightDif", "KODif", "SubDif", "WinDif", "LossDif",
    "WinStreakDif", "LoseStreakDif", "LongestWinStreakDif",
    "TotalRoundDif", "TotalTitleBoutDif",
    "wc_rank_diff", "pfp_rank_diff",
    "sig_str_pct_diff", "td_pct_diff",
]


# ============================================================================
# MAIN
# ============================================================================

def compute_stats(filepath: Path) -> dict:
    """Load CSV, time-split, compute stats on train portion."""
    print(f"  Reading: {filepath.name}")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    cut = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:cut]

    date_min = train["Date"].min().strftime("%Y-%m-%d")
    date_max = train["Date"].max().strftime("%Y-%m-%d")
    print(f"  Train: {len(train):,} rows ({date_min} to {date_max})")

    # Compute stats for each display feature present in the file
    available = [f for f in DISPLAY_FEATURES if f in train.columns]
    missing = [f for f in DISPLAY_FEATURES if f not in train.columns]
    if missing:
        print(f"  Skipped (not in file): {missing}")

    stats = {}
    for feat in available:
        col = train[feat].dropna()
        stats[feat] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "p05": float(np.percentile(col, 5)),
            "p95": float(np.percentile(col, 95)),
        }

    print(f"  Computed stats for {len(stats)} features")
    return {
        "source": str(filepath.relative_to(PROJECT_ROOT)),
        "train_rows": len(train),
        "date_range": {"min": date_min, "max": date_max},
        "stats": stats,
    }


def main():
    print("=" * 50)
    print("Building display normalization stats")
    print("=" * 50)

    result = {}
    for key, path in SOURCES.items():
        print(f"\n[{key}]")
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        result[key] = compute_stats(path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[SAVED] {OUTPUT_PATH}")
    print("[DONE]")


if __name__ == "__main__":
    main()
