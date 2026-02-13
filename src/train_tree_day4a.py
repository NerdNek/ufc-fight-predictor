"""
train_tree_day4a.py - Tree Model with Odds Interaction Features

Trains a HistGradientBoostingClassifier with curated interaction features
that learn conditional effects (e.g., "reach matters more when odds are close").

Improvements over Day 3:
- Interaction features for close-odds and favorite gates
- Segment evaluation (close-odds, underdogs, favorites)
- Non-linear decision boundaries via gradient boosting

Usage:
    python src/train_tree_day4a.py
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss,
    brier_score_loss,
)
from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = Path("data/processed/features_with_odds.csv")
MODEL_PATH = Path("models/hgb_day4a.joblib")
REPORT_PATH = Path("reports/day4a_metrics.json")
SEGMENT_PATH = Path("reports/day4a_segment_metrics.json")

# Columns to exclude from features
EXCLUDE_COLS = ["target", "RedFighter", "BlueFighter", "Date", "stance_matchup"]

# Train/test split ratio
TRAIN_RATIO = 0.80

# Interaction feature configuration
CLOSE_ODDS_THRESHOLD = 0.75  # abs(odds_diff) < this → "close odds"


# ============================================================================
# TIME-BASED SPLIT
# ============================================================================


def time_split(df: pd.DataFrame, date_col: str = "Date", train_frac: float = 0.8):
    """
    Split data by time (no shuffling).
    Train on oldest fights, test on newest fights.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    n = len(df)
    cut = int(n * train_frac)

    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()

    print(f"[SPLIT] Time-based split ({train_frac:.0%}/{1-train_frac:.0%})")
    print(f"   Train: {len(train_df):,} rows")
    print(f"   Test:  {len(test_df):,} rows")

    return train_df, test_df


# ============================================================================
# INTERACTION FEATURES
# ============================================================================


def build_interactions(df: pd.DataFrame, odds_col: str = "odds_diff", t: float = 0.75):
    """
    Adds interaction features to learn conditional effects.
    
    Gates:
    - close_odds: 1 if abs(odds_diff) < t, else 0
    - red_fav_gate: 1 if odds_diff > 0, else 0 (Red is market favorite)
    - red_underdog_gate: 1 if odds_diff < 0, else 0 (Red is underdog)
    
    Gate sign verified from Day 3 odds-only model coefficient:
    - odds_diff coef = +0.2496 (positive)
    - Higher odds_diff → higher P(Red wins)
    - Therefore: odds_diff > 0 → Red is favorite (correct gate sign)
    - Verified 2026-02-01.
    """
    out = df.copy()

    if odds_col not in out.columns:
        raise ValueError(f"Missing required odds column: {odds_col}")

    print(f"\n[INTERACT] Building interaction features (threshold={t})...")

    # ----------------------------------------------------------------
    # GATES
    # ----------------------------------------------------------------
    
    # Close-odds gate: odds are uncertain, skill matters more
    out["close_odds"] = (out[odds_col].abs() < t).astype(int)
    close_pct = out["close_odds"].mean() * 100
    print(f"   + close_odds gate: {close_pct:.1f}% of fights")

    # Red favorite gate (verified from Day 3 coefficient sign: +0.2496)
    out["red_fav_gate"] = (out[odds_col] > 0).astype(int)
    fav_pct = out["red_fav_gate"].mean() * 100
    print(f"   + red_fav_gate: {fav_pct:.1f}% Red favorites")

    # Red underdog gate (inverse of favorite)
    out["red_underdog_gate"] = (out[odds_col] < 0).astype(int)
    dog_pct = out["red_underdog_gate"].mean() * 100
    print(f"   + red_underdog_gate: {dog_pct:.1f}% Red underdogs")

    # ----------------------------------------------------------------
    # CLOSE-ODDS INTERACTIONS
    # "When odds are close, these skills matter more"
    # ----------------------------------------------------------------
    
    close_interactions = {
        "ReachDif": "reach_x_close",
        "AgeDif": "age_x_close",
        "AvgTDDif": "td_x_close",
        "SigStrDif": "sig_x_close",
        "AvgSubAttDif": "sub_x_close",
    }

    for base_col, new_col in close_interactions.items():
        if base_col in out.columns:
            out[new_col] = out[base_col] * out["close_odds"]
            print(f"   + {new_col} = {base_col} × close_odds")
        else:
            print(f"   ! Skipping {new_col}: {base_col} not found")

    # ----------------------------------------------------------------
    # FAVORITE INTERACTIONS
    # "When Red is favorite, these skills have different effects"
    # ----------------------------------------------------------------
    
    fav_interactions = {
        "AvgTDDif": "td_x_redfav",
        "ReachDif": "reach_x_redfav",
    }

    for base_col, new_col in fav_interactions.items():
        if base_col in out.columns:
            out[new_col] = out[base_col] * out["red_fav_gate"]
            print(f"   + {new_col} = {base_col} × red_fav_gate")

    # ----------------------------------------------------------------
    # UNDERDOG INTERACTIONS
    # "When Red is underdog, upset potential from these skills"
    # ----------------------------------------------------------------
    
    underdog_interactions = {
        "AvgTDDif": "td_x_underdog",
        "ReachDif": "reach_x_underdog",
        "SigStrDif": "sig_x_underdog",
    }

    for base_col, new_col in underdog_interactions.items():
        if base_col in out.columns:
            out[new_col] = out[base_col] * out["red_underdog_gate"]
            print(f"   + {new_col} = {base_col} × red_underdog_gate")

    return out


# ============================================================================
# FEATURE PREPARATION
# ============================================================================


def get_X_y(df: pd.DataFrame):
    """
    Extract features (X) and target (y) from dataframe.
    Ensures all features are numeric with no NaN.
    """
    df = df.copy()

    # Target
    if "target" in df.columns:
        y = df["target"].astype(int).values
    elif "Winner" in df.columns:
        y = (df["Winner"] == "Red").astype(int).values
    else:
        raise ValueError("No target column found (target or Winner)")

    # Feature columns (exclude non-features)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()

    # Convert boolean columns to int (pandas reads True/False from CSV as bool dtype)
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        for col in bool_cols:
            X[col] = X[col].astype(int)
        print(f"   Converted {len(bool_cols)} boolean columns to int")

    # Validate: all numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"   ! Dropping non-numeric columns: {non_numeric}")
        feature_cols = [c for c in feature_cols if c not in non_numeric]
        X = X[feature_cols]

    # Validate: no NaN
    nan_counts = X.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"   ! Filling {len(cols_with_nan)} columns with NaN -> 0")
        X = X.fillna(0)

    return X, y, feature_cols


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate(model, X_test, y_test):
    """Compute classification and calibration metrics."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }


def segment_eval(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    proba: np.ndarray,
    odds_col: str = "odds_diff",
    t: float = 0.75,
):
    """
    Evaluate model on meaningful segments with base rates.
    Includes:
    - N, red_win_rate, model_accuracy, majority_baseline, AUC
    """
    seg = {}

    if odds_col not in df_test.columns:
        print("   ! odds_diff not found, skipping segment eval")
        return seg

    odds = df_test[odds_col].values

    def eval_segment(mask, name):
        """Helper to compute segment metrics with base rate."""
        if mask.sum() < 10:
            return None
        
        y_seg = y_test[mask]
        p_seg = proba[mask]
        pred_seg = (p_seg >= 0.5).astype(int)
        
        # Base rate and majority baseline
        red_win_rate = float(y_seg.mean())
        majority_class = 1 if red_win_rate >= 0.5 else 0
        majority_baseline = float(max(red_win_rate, 1 - red_win_rate))
        
        try:
            auc = float(roc_auc_score(y_seg, p_seg))
        except ValueError:
            auc = None  # Single class
        
        return {
            "n": int(mask.sum()),
            "red_win_rate": red_win_rate,
            "model_accuracy": float(accuracy_score(y_seg, pred_seg)),
            "majority_baseline": majority_baseline,
            "lift_vs_baseline": float(accuracy_score(y_seg, pred_seg)) - majority_baseline,
            "roc_auc": auc,
        }

    # Close-odds segment
    close_mask = np.abs(odds) < t
    result = eval_segment(close_mask, "close_odds")
    if result:
        seg["close_odds"] = result

    # Not-close-odds segment
    far_mask = ~close_mask
    result = eval_segment(far_mask, "not_close_odds")
    if result:
        seg["not_close_odds"] = result

    # Red favorites segment (odds_diff > 0)
    fav_mask = odds > 0
    result = eval_segment(fav_mask, "red_favorites")
    if result:
        seg["red_favorites"] = result

    # Red underdogs segment (odds_diff < 0)
    dog_mask = odds < 0
    result = eval_segment(dog_mask, "red_underdogs")
    if result:
        seg["red_underdogs"] = result

    return seg


def threshold_sweep(
    df: pd.DataFrame,
    thresholds: list = [0.25, 0.5, 0.75, 1.0, 1.25],
    train_frac: float = 0.8,
):
    """
    Sweep close-odds thresholds to find optimal segmentation.
    Returns results for each threshold.
    """
    print("\n[SWEEP] Threshold sensitivity analysis...")
    results = []
    
    train_df, test_df = time_split(df, date_col="Date", train_frac=train_frac)
    
    for t in thresholds:
        # Build interactions with this threshold
        train_t = build_interactions(train_df.copy(), odds_col="odds_diff", t=t)
        test_t = build_interactions(test_df.copy(), odds_col="odds_diff", t=t)
        
        X_train, y_train, feat_cols = get_X_y(train_t)
        X_test, y_test, _ = get_X_y(test_t)
        
        # Train model
        model = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, max_iter=300,
            min_samples_leaf=25, random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        proba = model.predict_proba(X_test)[:, 1]
        seg = segment_eval(test_t, y_test, proba, odds_col="odds_diff", t=t)
        
        close_n = seg.get("close_odds", {}).get("n", 0)
        close_auc = seg.get("close_odds", {}).get("roc_auc", None)
        close_lift = seg.get("close_odds", {}).get("lift_vs_baseline", None)
        
        results.append({
            "threshold": t,
            "close_n": close_n,
            "close_auc": close_auc,
            "close_lift": close_lift,
            "overall_auc": float(roc_auc_score(y_test, proba)),
        })
        
        print(f"   t={t:.2f}: close_n={close_n:4d}, close_auc={close_auc if close_auc else 'N/A':>6}, overall_auc={results[-1]['overall_auc']:.4f}")
    
    return results


def ablation_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    odds_features: list,
    skill_features: list,
):
    """
    Compare model performance with different feature sets:
    - odds_only: just betting odds
    - skill_only: no odds features
    - full: all features
    """
    print("\n[ABLATION] Feature set comparison...")
    results = {}
    
    X_train_full, y_train, all_cols = get_X_y(train_df)
    X_test_full, y_test, _ = get_X_y(test_df)
    
    # Identify available columns in each category
    odds_cols = [c for c in odds_features if c in X_train_full.columns]
    skill_cols = [c for c in skill_features if c in X_train_full.columns]
    
    print(f"   Odds features available: {len(odds_cols)}")
    print(f"   Skill features available: {len(skill_cols)}")
    
    def train_eval(X_tr, X_te, name):
        model = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, max_iter=300,
            min_samples_leaf=25, random_state=42
        )
        model.fit(X_tr, y_train)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        
        return {
            "model": name,
            "n_features": X_tr.shape[1],
            "accuracy": float(accuracy_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "log_loss": float(log_loss(y_test, proba)),
            "brier_score": float(brier_score_loss(y_test, proba)),
        }
    
    # Odds-only tree
    if odds_cols:
        results["odds_only_tree"] = train_eval(
            X_train_full[odds_cols], X_test_full[odds_cols], "HGB Odds-Only"
        )
        print(f"   Odds-only: AUC={results['odds_only_tree']['roc_auc']:.4f}")
    
    # Skill-only tree (no odds)
    if skill_cols:
        results["skill_only_tree"] = train_eval(
            X_train_full[skill_cols], X_test_full[skill_cols], "HGB Skill-Only"
        )
        print(f"   Skill-only: AUC={results['skill_only_tree']['roc_auc']:.4f}")
    
    # Full tree
    results["full_tree"] = train_eval(
        X_train_full, X_test_full, "HGB Full"
    )
    print(f"   Full model: AUC={results['full_tree']['roc_auc']:.4f}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 60)
    print("UFC Fight Predictor - Day 4A: Tree Model with Interactions")
    print("=" * 60)

    # Ensure output directories exist
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n[LOAD] Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Time-based split (before adding interactions to avoid leakage)
    train_df, test_df = time_split(df, date_col="Date", train_frac=TRAIN_RATIO)

    # Add interaction features
    train_df = build_interactions(train_df, odds_col="odds_diff", t=CLOSE_ODDS_THRESHOLD)
    test_df = build_interactions(test_df, odds_col="odds_diff", t=CLOSE_ODDS_THRESHOLD)

    # Prepare X, y
    print("\n[PREP] Preparing features and target...")
    X_train, y_train, feature_cols = get_X_y(train_df)
    X_test, y_test, _ = get_X_y(test_df)
    print(f"   Features: {len(feature_cols)} columns")
    print(f"   Train y distribution: {y_train.mean()*100:.1f}% Red wins")
    print(f"   Test y distribution:  {y_test.mean()*100:.1f}% Red wins")

    # Train model
    print("\n[TRAIN] Training HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        min_samples_leaf=25,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("   Training complete.")

    # Evaluate
    print("\n[EVAL] Evaluating on test set...")
    metrics = evaluate(model, X_test, y_test)
    print(f"   Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"   Log Loss:    {metrics['log_loss']:.4f}")
    print(f"   Brier Score: {metrics['brier_score']:.4f}")
    print(f"\n   Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"                 Predicted")
    print(f"                 Blue    Red")
    print(f"   Actual Blue   {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"   Actual Red    {cm[1][0]:4d}   {cm[1][1]:4d}")

    # Segment evaluation with base rates
    print("\n[SEGMENT] Evaluating on subsets (with base rates)...")
    proba = model.predict_proba(X_test)[:, 1]
    seg = segment_eval(test_df, y_test, proba, odds_col="odds_diff", t=CLOSE_ODDS_THRESHOLD)

    print(f"\n   {'Segment':<16} {'N':>6} {'RedWin%':>8} {'Acc':>7} {'Baseline':>9} {'Lift':>7} {'AUC':>7}")
    print("   " + "-" * 62)
    for seg_name, seg_data in seg.items():
        if seg_data.get("roc_auc") is not None:
            print(
                f"   {seg_name:<16} {seg_data['n']:>6} "
                f"{seg_data['red_win_rate']*100:>7.1f}% "
                f"{seg_data['model_accuracy']:.4f} "
                f"{seg_data['majority_baseline']:.4f} "
                f"{seg_data['lift_vs_baseline']:>+6.3f} "
                f"{seg_data['roc_auc']:.4f}"
            )
        else:
            print(f"   {seg_name:<16} {seg_data['n']:>6} (single class)")

    # Ablation study
    odds_features = ["odds_diff", "ev_diff", "dec_odds_diff", "sub_odds_diff", "ko_odds_diff"]
    skill_features = [
        "ReachDif", "AgeDif", "AvgTDDif", "SigStrDif", "AvgSubAttDif",
        "HeightDif", "WinDif", "LossDif", "KODif", "SubDif",
        "WinStreakDif", "LoseStreakDif", "LongestWinStreakDif",
        "TotalRoundDif", "TotalTitleBoutDif",
        "sig_str_pct_diff", "td_pct_diff",
    ]
    ablation = ablation_study(train_df, test_df, odds_features, skill_features)

    # Save model
    print(f"\n[SAVE] Saving model to: {MODEL_PATH}")
    dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)

    # Save metrics
    payload = {
        "timestamp": datetime.now().isoformat(),
        "model": "HistGradientBoostingClassifier",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(feature_cols)),
        "interaction_gate_threshold": CLOSE_ODDS_THRESHOLD,
        "metrics": metrics,
        "ablation": ablation,
        "baselines_comparison": {
            "majority_class_accuracy": 0.5628,
            "odds_only_roc_auc": 0.7229,
            "logreg_day3_roc_auc": 0.6988,
        },
    }

    print(f"[SAVE] Saving metrics to: {REPORT_PATH}")
    REPORT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"[SAVE] Saving segment metrics to: {SEGMENT_PATH}")
    SEGMENT_PATH.write_text(json.dumps(seg, indent=2))

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<35} {'Accuracy':>10} {'ROC-AUC':>10}")
    print("-" * 57)
    print(f"{'Majority Class':<35} {'56.28%':>10} {'N/A':>10}")
    print(f"{'Odds Only (LogReg Day 3)':<35} {'66.85%':>10} {'0.7229':>10}")
    print(f"{'Full LogReg Day 3':<35} {'62.94%':>10} {'0.6988':>10}")
    
    # Ablation results
    if "odds_only_tree" in ablation:
        a = ablation["odds_only_tree"]
        print(f"{'HGB Odds-Only':<35} {a['accuracy']*100:>9.2f}% {a['roc_auc']:>10.4f}")
    if "skill_only_tree" in ablation:
        a = ablation["skill_only_tree"]
        print(f"{'HGB Skill-Only':<35} {a['accuracy']*100:>9.2f}% {a['roc_auc']:>10.4f}")
    
    print(
        f"{'HGB Full (' + str(len(feature_cols)) + ' features)':<35} "
        f"{metrics['accuracy']*100:>9.2f}% {metrics['roc_auc']:>10.4f}"
    )
    print("-" * 57)

    # Success checks
    print("\n[CHECK] Success criteria:")
    if metrics["roc_auc"] >= 0.7229:
        print("   [OK] PRIMARY: Beat odds-only baseline (ROC-AUC >= 0.723)")
    elif metrics["roc_auc"] >= 0.6988:
        print("   [**] SECONDARY: Beat LogReg Day 3 (ROC-AUC >= 0.699)")
    else:
        print("   [!!] Did not beat Day 3 baseline overall")

    # Check segment lift (with base rate validation)
    for seg_name, seg_data in seg.items():
        lift = seg_data.get("lift_vs_baseline", 0)
        if lift > 0.02:  # >2% lift is meaningful
            print(f"   [OK] +{lift*100:.1f}% lift vs baseline in {seg_name}")

    print("\n" + "=" * 60)
    print("[DONE] Day 4A complete!")
    print("=" * 60)

    return metrics, seg, ablation


if __name__ == "__main__":
    main()
