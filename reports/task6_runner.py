"""
Task 6: Regression verification on 3 cases.
  1. Holloway vs Oliveira (Mode A historical + forced Mode B synthetic)
  2. One-sided matchup
  3. Close/toss-up matchup

Output -> reports/day6_synthetic_sanity_runs.txt
"""
import sys
sys.path.insert(0, ".")

from src.predict import (
    _load_artifacts,
    get_fighter_profile,
    build_synthetic_row,
    enforce_schema_lock,
    check_ood_features,
    predict_matchup,
)

lines = []
arts = _load_artifacts()

def section(title):
    lines.append("")
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)

def run_synthetic(red, blue, label, arts):
    """Force Mode B for any fighter pair."""
    model = arts["model_no_odds"]
    model_cols = arts["model_cols_no_odds"]
    cleaned_df = arts["cleaned_df"]

    red_profile = get_fighter_profile(cleaned_df, red)
    blue_profile = get_fighter_profile(cleaned_df, blue)
    row_df = build_synthetic_row(red_profile, blue_profile)
    bool_cols = row_df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        row_df[col] = row_df[col].astype(int)
    row_df = row_df.fillna(0)

    # Before OOD guardrail
    key_feats = ["SigStrDif", "AvgTDDif", "AvgSubAttDif", "SubDif", "KODif"]
    before = {f: row_df[f].iloc[0] for f in key_feats if f in row_df.columns}

    # Apply OOD guardrail
    row_df = check_ood_features(row_df)
    after = {f: row_df[f].iloc[0] for f in key_feats if f in row_df.columns}

    X = enforce_schema_lock(row_df, model_cols)
    proba = model.predict_proba(X)[0][1]
    winner = red if proba >= 0.5 else blue

    lines.append(f"  [{label}] {red} vs {blue}")
    lines.append(f"  Mode: SYNTHETIC (forced)")
    lines.append(f"  P(Red wins) = {proba:.4f}")
    lines.append(f"  Predicted winner: {winner}")
    lines.append(f"  Red data from: {red_profile['date']}")
    lines.append(f"  Blue data from: {blue_profile['date']}")
    lines.append("")
    lines.append("  Key features (before -> after OOD clamp):")
    for f in key_feats:
        b = before.get(f)
        a = after.get(f)
        if b is not None:
            clipped = "(CLIPPED)" if abs(b - a) > 0.001 else ""
            lines.append(f"    {f:20s}  {b:>8.2f} -> {a:>8.2f}  {clipped}")
    lines.append("")

# ============================
# 1. Holloway vs Oliveira
# ============================
section("Case 1: Max Holloway vs Charles Oliveira (previously suspicious)")

# Mode A (historical)
r1 = predict_matchup("Max Holloway", "Charles Oliveira")
lines.append(f"  [Mode A - Historical]")
lines.append(f"  P(Red wins) = {r1['proba_red']:.4f}")
lines.append(f"  Based on fight: {r1['fight_date']}")
lines.append(f"  Diffs: {r1['diffs']}")
lines.append("")

# Mode B (forced synthetic)
run_synthetic("Max Holloway", "Charles Oliveira", "Mode B - Synthetic", arts)


# ============================
# 2. One-sided matchup (strong favorite)
# ============================
section("Case 2: Jon Jones vs Tai Tuivasa (one-sided)")
run_synthetic("Jon Jones", "Tai Tuivasa", "Mode B - Synthetic", arts)


# ============================
# 3. Close/toss-up matchup
# ============================
section("Case 3: Dustin Poirier vs Justin Gaethje (close/toss-up)")
run_synthetic("Dustin Poirier", "Justin Gaethje", "Mode B - Synthetic", arts)


output = "\n".join(lines)
print(output)

out_path = "reports/day6_synthetic_sanity_runs.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output)
print(f"\n[SAVED] {out_path}")
