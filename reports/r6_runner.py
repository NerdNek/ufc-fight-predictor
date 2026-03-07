"""
Task R6: V2 synthetic sanity runs.
Compare v1 vs v2 synthetic predictions for 3 cases.
Output -> reports/v2_synthetic_sanity_runs.txt
"""
import sys
sys.path.insert(0, ".")

from src.predict import predict_matchup, _load_artifacts, get_fighter_profile, build_synthetic_row, enforce_schema_lock, check_ood_features

lines = []
arts = _load_artifacts()

def section(title):
    lines.append("")
    lines.append("=" * 70)
    lines.append(title)
    lines.append("=" * 70)

def run_case(red, blue, label):
    """Run prediction and capture key diffs."""
    result = predict_matchup(red, blue)
    mode = result["mode"]
    p_red = result["proba_red"]
    diffs = result.get("diffs", {})

    lines.append(f"  [{label}] {red} vs {blue}")
    lines.append(f"  Mode:            {mode}")
    lines.append(f"  P(Red wins):     {p_red:.4f} ({p_red*100:.1f}%)")
    lines.append(f"  Predicted:       {result['winner_name']}")
    if "historical" in mode:
        lines.append(f"  Fight date:      {result.get('fight_date', '?')}")
    else:
        lines.append(f"  Red data from:   {result.get('red_data_from', '?')}")
        lines.append(f"  Blue data from:  {result.get('blue_data_from', '?')}")
    lines.append("")
    lines.append("  Key diffs driving prediction:")
    for k, v in diffs.items():
        lines.append(f"    {k:20s}  {v:>8.2f}" if isinstance(v, float) else f"    {k:20s}  {v}")
    lines.append("")

    # Also force synthetic for comparison
    if "historical" in mode:
        lines.append("  [Forced Synthetic v2 comparison]")
        model = arts["model_no_odds"]
        model_cols = arts["model_cols_no_odds"]
        cd = arts["cleaned_df"]
        rp = get_fighter_profile(cd, red)
        bp = get_fighter_profile(cd, blue)
        row = build_synthetic_row(rp, bp)
        for c in row.select_dtypes(include=["bool"]).columns:
            row[c] = row[c].astype(int)
        row = row.fillna(0)
        row = check_ood_features(row)
        X = enforce_schema_lock(row, model_cols)
        p = model.predict_proba(X)[0][1]
        lines.append(f"  Synthetic v2 P(Red): {p:.4f} ({p*100:.1f}%)")
        key_feats = ["SigStrDif", "AvgTDDif", "AvgSubAttDif", "SubDif", "KODif"]
        for f in key_feats:
            if f in row.columns:
                lines.append(f"    {f:20s}  {row[f].iloc[0]:>8.2f}")
        lines.append("")

# Header
lines.append("V2 Synthetic Sanity Run Report")
lines.append(f"Model: hgb_no_odds_v2.joblib (Mode B)")
lines.append(f"Historical model: hgb_day4a.joblib (Mode A, unchanged)")

# Case 1
section("Case 1: Max Holloway vs Charles Oliveira (previously suspicious)")
run_case("Max Holloway", "Charles Oliveira", "Primary")

# Case 2
section("Case 2: Jon Jones vs Tai Tuivasa (one-sided)")
run_case("Jon Jones", "Tai Tuivasa", "One-sided")

# Case 3
section("Case 3: Dustin Poirier vs Justin Gaethje (close/toss-up)")
run_case("Dustin Poirier", "Justin Gaethje", "Toss-up")

output = "\n".join(lines)
print(output)

with open("reports/v2_synthetic_sanity_runs.txt", "w", encoding="utf-8") as f:
    f.write(output)
print("\n[SAVED] reports/v2_synthetic_sanity_runs.txt")
