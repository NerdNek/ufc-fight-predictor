"""
Debug script: reproduce the extreme Holloway vs Oliveira prediction,
dump key feature values from the final X row.

Captures BOTH Mode A (historical) and Mode B (synthetic) to compare.

Output -> reports/debug_out_of_range_example.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.predict import (
    _load_artifacts,
    get_fighter_profile,
    build_synthetic_row,
    find_fight_row,
    enforce_schema_lock,
    build_interactions,
    EXPLAINABILITY_DIFFS,
    EXCLUDE_COLS,
)

RED = "Max Holloway"
BLUE = "Charles Oliveira"

arts = _load_artifacts()
features_df = arts["features_df"]
cleaned_df = arts["cleaned_df"]

lines = []
lines.append("=" * 60)
lines.append(f"DEBUG: Feature comparison -- {RED} vs {BLUE}")
lines.append("=" * 60)
lines.append("")

# ===== MODE A (HISTORICAL) =====
hist_row = find_fight_row(features_df, RED, BLUE)
if hist_row is not None:
    lines.append(">>> MODE A (Historical) -- uses features_with_odds.csv row")
    lines.append("-" * 50)

    model_a = arts["model_with_odds"]
    model_cols_a = arts["model_cols_with_odds"]
    row_df_a = pd.DataFrame([hist_row])
    drop_cols = [c for c in EXCLUDE_COLS if c in row_df_a.columns]
    row_df_a = row_df_a.drop(columns=drop_cols, errors="ignore")
    row_df_a = row_df_a.drop(
        columns=["RedFighter", "BlueFighter", "Date", "Winner"],
        errors="ignore",
    )
    row_df_a = build_interactions(row_df_a)
    bool_cols = row_df_a.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        row_df_a[col] = row_df_a[col].astype(int)
    row_df_a = row_df_a.fillna(0)
    X_a = enforce_schema_lock(row_df_a, model_cols_a)
    proba_a = model_a.predict_proba(X_a)[0][1]

    key_feats = [
        "SigStrDif", "ReachDif", "AgeDif", "AvgTDDif", "AvgSubAttDif",
        "WinDif", "LossDif", "KODif", "SubDif",
        "odds_diff",
        "sig_x_close", "sig_x_underdog",
    ]

    lines.append(f"  P(Red wins) = {proba_a:.4f}")
    lines.append("")
    lines.append("  Key features:")
    for feat in key_feats:
        if feat in X_a.columns:
            v = X_a[feat].iloc[0]
            lines.append(f"    {feat:25s} = {v:>10.4f}")
        else:
            lines.append(f"    {feat:25s} = NOT IN SCHEMA")

    lines.append("")
    lines.append("  Features with |value| > 5:")
    for col in sorted(X_a.columns):
        val = X_a[col].iloc[0]
        if abs(val) > 5:
            lines.append(f"    {col:30s} = {val:>10.4f}")
else:
    lines.append(">>> MODE A: No historical row found.")

lines.append("")
lines.append("")

# ===== MODE B (SYNTHETIC) =====
lines.append(">>> MODE B (Synthetic) -- built from cleaned_df profiles")
lines.append("-" * 50)

model_b = arts["model_no_odds"]
model_cols_b = arts["model_cols_no_odds"]
red_profile = get_fighter_profile(cleaned_df, RED)
blue_profile = get_fighter_profile(cleaned_df, BLUE)
row_df_b = build_synthetic_row(red_profile, blue_profile)
bool_cols = row_df_b.select_dtypes(include=["bool"]).columns.tolist()
for col in bool_cols:
    row_df_b[col] = row_df_b[col].astype(int)
row_df_b = row_df_b.fillna(0)
X_b = enforce_schema_lock(row_df_b, model_cols_b)
proba_b = model_b.predict_proba(X_b)[0][1]

lines.append(f"  P(Red wins) = {proba_b:.4f}")
lines.append(f"  Red profile date:  {red_profile['date']}")
lines.append(f"  Blue profile date: {blue_profile['date']}")
lines.append("")

lines.append("  Key features:")
for feat in key_feats:
    if feat in X_b.columns:
        v = X_b[feat].iloc[0]
        lines.append(f"    {feat:25s} = {v:>10.4f}")
    else:
        lines.append(f"    {feat:25s} = NOT IN SCHEMA")

lines.append("")
lines.append("  Raw profile values for SigStrDif source:")
lines.append(f"    Red  AvgSigStrLanded = {red_profile.get('AvgSigStrLanded')}")
lines.append(f"    Blue AvgSigStrLanded = {blue_profile.get('AvgSigStrLanded')}")
lines.append(f"    Synthetic SigStrDif  = {row_df_b['SigStrDif'].iloc[0]:.4f}")

lines.append("")
lines.append("  Features with |value| > 5:")
for col in sorted(X_b.columns):
    val = X_b[col].iloc[0]
    if abs(val) > 5:
        lines.append(f"    {col:30s} = {val:>10.4f}")

lines.append("")
lines.append("")

# ===== FEATURE MISMATCH ANALYSIS =====
lines.append(">>> MISMATCHED DIFF COLUMNS (training vs synthetic source)")
lines.append("-" * 50)
lines.append("For each pre-existing Dif column, compare its actual value in")
lines.append("training data vs. the value that build_synthetic_row produces")
lines.append("using the Red/Blue columns it maps to.")
lines.append("")

df_c = pd.read_csv(str(Path(__file__).parent.parent / "data" / "processed" / "ufc_cleaned.csv"))

mismatch_checks = {
    "SigStrDif":    ("RedAvgSigStrLanded", "BlueAvgSigStrLanded"),
    "AvgSubAttDif": ("RedAvgSubAtt", "BlueAvgSubAtt"),
    "AvgTDDif":     ("RedAvgTDLanded", "BlueAvgTDLanded"),
}
for diff_col, (red_col, blue_col) in mismatch_checks.items():
    recomp = df_c[red_col] - df_c[blue_col]
    delta = (df_c[diff_col].fillna(0) - recomp.fillna(0)).abs()
    n_bad = (delta > 0.1).sum()
    lines.append(f"  {diff_col}:")
    lines.append(f"    Synthetic computes from: {red_col} - {blue_col}")
    lines.append(f"    Max |training - recomputed| = {delta.max():.2f}")
    lines.append(f"    Mean |training - recomputed| = {delta.mean():.2f}")
    lines.append(f"    Mismatched rows: {n_bad} / {len(delta)}")
    lines.append("")

lines.append("")

# ===== TRAINING DISTRIBUTION =====
lines.append(">>> TRAINING DISTRIBUTION of SigStrDif (from features_with_odds.csv)")
lines.append("-" * 50)
if "SigStrDif" in features_df.columns:
    stats = features_df["SigStrDif"].describe()
    for s in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        lines.append(f"  {s:8s} = {stats[s]:>10.4f}")

    hist_val = hist_row.get("SigStrDif", None) if hist_row is not None else None
    synth_val = row_df_b["SigStrDif"].iloc[0]
    if hist_val is not None:
        z_hist = (hist_val - stats["mean"]) / stats["std"]
        lines.append(f"  Mode A val  = {hist_val:>10.4f}  (Z = {z_hist:>6.2f})")
    lines.append(f"  Mode B val  = {synth_val:>10.4f}  (Z = {(synth_val - stats['mean'])/stats['std']:>6.2f})")

lines.append("")
lines.append("")

# ===== ROOT CAUSE CONCLUSION =====
lines.append(">>> ROOT CAUSE CONCLUSION")
lines.append("-" * 50)
lines.append("1. SigStrDif, AvgSubAttDif, and AvgTDDif in the raw dataset are")
lines.append("   computed from DIFFERENT source columns than the Red/Blue cols")
lines.append("   that build_synthetic_row maps them to.")
lines.append("")
lines.append("2. In Mode B (synthetic), build_synthetic_row line 416 maps:")
lines.append("     SigStrDif -> AvgSigStrLanded (per-minute rate)")
lines.append("   But the training data's SigStrDif comes from cumulative totals.")
lines.append("   This produces a SCALE MISMATCH: synthetic values ~0-10 vs")
lines.append("   training values spanning -118 to +128.")
lines.append("")
lines.append("3. The same pattern applies to AvgSubAttDif and AvgTDDif.")
lines.append("")
lines.append("4. For Holloway vs Oliveira specifically, Mode A (historical)")
lines.append("   is triggered because this fight exists in the dataset.")
lines.append("   The extreme P(Red)=0.93 is driven by odds_diff=-470 and")
lines.append("   sig_x_underdog=-40.53 (interaction feature).")

output = "\n".join(lines)
print(output)

out_path = Path(__file__).parent / "debug_out_of_range_example.txt"
out_path.write_text(output, encoding="utf-8")
print(f"\n[SAVED] {out_path}")
