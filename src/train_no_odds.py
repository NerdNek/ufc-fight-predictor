"""
train_no_odds.py - Train HGB Model Without Odds Features

Trains a HistGradientBoostingClassifier on the no-odds feature set.
This model is used for synthetic/upcoming matchups where betting odds
are unavailable.

Usage:
    python src/train_no_odds.py
    python src/train_no_odds.py --input data/processed/features_no_odds_v2.csv --out-suffix _v2
"""


import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = Path("data/processed/features_no_odds.csv")
MODEL_PATH = Path("models/hgb_no_odds.joblib")
REPORT_PATH = Path("reports/hgb_no_odds_metrics.json")

EXCLUDE_COLS = ["target", "RedFighter", "BlueFighter", "Date", "stance_matchup"]
TRAIN_RATIO = 0.80


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train HGB no-odds model.")
    parser.add_argument(
        "--input", default=None,
        help="Path to features CSV (default: data/processed/features_no_odds.csv)"
    )
    parser.add_argument(
        "--out-suffix", default="",
        help="Suffix for output filenames, e.g. _v2"
    )
    args = parser.parse_args()

    sfx = args.out_suffix
    data_path = Path(args.input) if args.input else DATA_PATH
    model_path = Path(f"models/hgb_no_odds{sfx}.joblib")
    report_path = Path(f"reports/hgb_no_odds_metrics{sfx}.json")

    print("=" * 60)
    print("UFC Fight Predictor - HGB No-Odds Model")
    print("=" * 60)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n[LOAD] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Time-based split
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    n = len(df)
    cut = int(n * TRAIN_RATIO)
    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()
    print(f"[SPLIT] Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")

    # Prepare X, y
    def get_X_y(split_df):
        y = split_df["target"].astype(int).values
        feature_cols = [c for c in split_df.columns if c not in EXCLUDE_COLS]
        X = split_df[feature_cols].copy()

        # Convert booleans to int
        bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
        for col in bool_cols:
            X[col] = X[col].astype(int)

        # Drop non-numeric
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            feature_cols = [c for c in feature_cols if c not in non_numeric]
            X = X[feature_cols]

        X = X.fillna(0)
        return X, y, feature_cols

    X_train, y_train, feature_cols = get_X_y(train_df)
    X_test, y_test, _ = get_X_y(test_df)
    print(f"[PREP] Features: {len(feature_cols)} columns (no odds, no interactions)")

    # Train model (same hyperparameters as with-odds HGB)
    print("\n[TRAIN] Training HistGradientBoostingClassifier (no-odds)...")
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
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
    }

    print(f"\n[EVAL] Test set results:")
    print(f"   Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"   Log Loss:    {metrics['log_loss']:.4f}")
    print(f"   Brier Score: {metrics['brier_score']:.4f}")

    # Save model
    print(f"\n[SAVE] Saving model to: {model_path}")
    dump({"model": model, "feature_cols": feature_cols}, model_path)

    # Save metrics
    payload = {
        "timestamp": datetime.now().isoformat(),
        "model": "HistGradientBoostingClassifier (no-odds)",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "metrics": metrics,
    }
    print(f"[SAVE] Saving metrics to: {report_path}")
    report_path.write_text(json.dumps(payload, indent=2))

    print(f"\n[DONE] No-odds HGB model trained with {len(feature_cols)} features")
    print(f"   Model: {model_path}")
    return metrics


if __name__ == "__main__":
    main()
