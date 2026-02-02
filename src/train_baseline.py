"""
train_baseline.py - Baseline Model Training with Time-Based Evaluation

Trains a baseline logistic regression model using time-based train/test split.
Compares against majority class and odds-only baselines.

Usage:
    python src/train_baseline.py
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix,
    classification_report
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature columns to exclude from X
EXCLUDE_COLS = ['target', 'RedFighter', 'BlueFighter', 'Date', 'stance_matchup']

# Train/test split ratio
TRAIN_RATIO = 0.80

# Random state for reproducibility
RANDOM_STATE = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath: Path) -> pd.DataFrame:
    """Load features dataset."""
    print(f"[LOAD] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare X (features) and y (target) from dataframe."""
    print("\n[PREP] Preparing features and target...")
    
    # Define feature columns (exclude non-features)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"   Features (X): {X.shape[1]} columns")
    print(f"   Target (y): {y.sum()} Red wins ({y.mean()*100:.1f}%), {(~y.astype(bool)).sum()} Blue wins")
    
    # Check for missing values
    missing = X.isnull().sum().sum()
    print(f"   Missing values: {missing}")
    
    return X, y, feature_cols


# ============================================================================
# TIME-BASED SPLIT
# ============================================================================

def time_based_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, 
                     train_ratio: float = 0.80) -> tuple:
    """
    Split data by time (no shuffling).
    Train on oldest fights, test on newest fights.
    This mimics real-world prediction scenario.
    """
    print(f"\n[SPLIT] Time-based train/test split ({train_ratio:.0%}/{1-train_ratio:.0%})...")
    
    # Parse dates
    dates = pd.to_datetime(df['Date'], errors='coerce')
    null_dates = dates.isna().sum()
    if null_dates > 0:
        print(f"   WARNING: {null_dates} rows with invalid dates (will be dropped)")
    
    # Sort by date
    sort_idx = dates.argsort()
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)
    dates_sorted = dates.iloc[sort_idx].reset_index(drop=True)
    
    # Split
    split_idx = int(len(X_sorted) * train_ratio)
    
    X_train = X_sorted.iloc[:split_idx]
    X_test = X_sorted.iloc[split_idx:]
    y_train = y_sorted.iloc[:split_idx]
    y_test = y_sorted.iloc[split_idx:]
    
    # Date ranges
    train_start = dates_sorted.iloc[0].strftime('%Y-%m-%d')
    train_end = dates_sorted.iloc[split_idx-1].strftime('%Y-%m-%d')
    test_start = dates_sorted.iloc[split_idx].strftime('%Y-%m-%d')
    test_end = dates_sorted.iloc[-1].strftime('%Y-%m-%d')
    
    print(f"   Train: {len(X_train):,} rows ({train_start} to {train_end})")
    print(f"   Test:  {len(X_test):,} rows ({test_start} to {test_end})")
    print(f"   Train y distribution: {y_train.mean()*100:.1f}% Red wins")
    print(f"   Test y distribution:  {y_test.mean()*100:.1f}% Red wins")
    
    split_info = {
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'train_date_range': f"{train_start} to {train_end}",
        'test_date_range': f"{test_start} to {test_end}",
        'train_red_win_rate': float(y_train.mean()),
        'test_red_win_rate': float(y_test.mean())
    }
    
    return X_train, X_test, y_train, y_test, split_info


# ============================================================================
# BASELINE MODELS
# ============================================================================

def majority_class_baseline(y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Majority class baseline: always predict the most common class in training.
    """
    print("\n[BASELINE 1] Majority Class Baseline...")
    
    # Find majority class in training
    majority_class = int(y_train.mode().iloc[0])
    majority_label = "Red" if majority_class == 1 else "Blue"
    
    # Predict majority class for all test samples
    y_pred = np.full(len(y_test), majority_class)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Majority class in train: {majority_label} ({majority_class})")
    print(f"   Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'name': 'Majority Class',
        'majority_class': majority_label,
        'accuracy': float(accuracy)
    }


def odds_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Odds-only baseline: logistic regression using only odds_diff.
    This is a strong baseline because betting markets are efficient.
    """
    print("\n[BASELINE 2] Odds-Only Baseline (single feature)...")
    
    # Use only odds_diff
    if 'odds_diff' not in X_train.columns:
        print("   WARNING: odds_diff not found, skipping odds baseline")
        return None
    
    X_train_odds = X_train[['odds_diff']].values
    X_test_odds = X_test[['odds_diff']].values
    
    # Train simple logistic regression
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train_odds, y_train)
    
    # Predict
    y_pred = model.predict(X_test_odds)
    y_proba = model.predict_proba(X_test_odds)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"   Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Test ROC-AUC:  {roc_auc:.4f}")
    
    return {
        'name': 'Odds Only (LogReg)',
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc)
    }


# ============================================================================
# MAIN MODEL
# ============================================================================

def train_logistic_regression(X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: pd.Series, y_test: pd.Series,
                               feature_cols: list) -> tuple:
    """
    Train logistic regression with all features.
    """
    print("\n[MODEL] Training Logistic Regression (all features)...")
    
    # Standardize features
    print("   Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("   Training model...")
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced'  # Handle mild imbalance
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Blue', 'Red'])
    
    print(f"\n   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Test ROC-AUC:  {roc_auc:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Blue    Red")
    print(f"   Actual Blue   {conf_matrix[0,0]:4d}   {conf_matrix[0,1]:4d}")
    print(f"   Actual Red    {conf_matrix[1,0]:4d}   {conf_matrix[1,1]:4d}")
    print(f"\n   Classification Report:")
    print(class_report)
    
    # Top features by coefficient magnitude
    coef_abs = np.abs(model.coef_[0])
    top_indices = np.argsort(coef_abs)[-10:][::-1]
    print("   Top 10 features by importance:")
    for i, idx in enumerate(top_indices):
        print(f"      {i+1}. {feature_cols[idx]}: {model.coef_[0][idx]:.4f}")
    
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }
    
    return model, scaler, metrics


# ============================================================================
# SAVE ARTIFACTS
# ============================================================================

def save_artifacts(model, scaler, metrics: dict, baselines: list, 
                   split_info: dict, feature_cols: list, project_root: Path):
    """Save model and metrics to disk."""
    print("\n[SAVE] Saving artifacts...")
    
    models_dir = project_root / 'models'
    reports_dir = project_root / 'reports'
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / 'logreg_baseline.joblib'
    joblib.dump({'model': model, 'scaler': scaler, 'feature_cols': feature_cols}, model_path)
    print(f"   Model saved: {model_path}")
    
    # Compile full results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'LogisticRegression',
        'split_info': split_info,
        'baselines': baselines,
        'model_metrics': metrics,
        'feature_count': len(feature_cols),
        'beat_majority_baseline': metrics['accuracy'] > baselines[0]['accuracy'],
        'beat_odds_baseline': metrics['roc_auc'] > baselines[1]['roc_auc'] if baselines[1] else None
    }
    
    # Save metrics
    metrics_path = reports_dir / 'baseline_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Metrics saved: {metrics_path}")
    
    return results


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def print_summary(results: dict):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    baselines = results['baselines']
    model = results['model_metrics']
    
    print(f"\n{'Model':<30} {'Accuracy':>10} {'ROC-AUC':>10}")
    print("-" * 52)
    
    for b in baselines:
        roc = f"{b.get('roc_auc', 'N/A'):.4f}" if b.get('roc_auc') else "N/A"
        print(f"{b['name']:<30} {b['accuracy']:>10.4f} {roc:>10}")
    
    print(f"{'Logistic Regression (Full)':<30} {model['accuracy']:>10.4f} {model['roc_auc']:>10.4f}")
    
    print("\n" + "-" * 52)
    if results['beat_majority_baseline']:
        print("[OK] Beat majority baseline")
    else:
        print("[!!] Did NOT beat majority baseline")
    
    if results['beat_odds_baseline']:
        print("[OK] Beat odds-only baseline")
    elif results['beat_odds_baseline'] is False:
        print("[!!] Did NOT beat odds-only baseline")
    
    if model['roc_auc'] > 0.60:
        print("[OK] ROC-AUC > 0.60 milestone reached!")
    else:
        print("[!!] ROC-AUC < 0.60 (more work needed)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'data' / 'processed' / 'ufc_features.csv'
    
    print("=" * 60)
    print("UFC Fight Predictor - Baseline Model Training")
    print("=" * 60)
    
    # 1. Load data
    df = load_data(input_path)
    
    # 2. Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # 3. Time-based split
    X_train, X_test, y_train, y_test, split_info = time_based_split(
        df, X, y, train_ratio=TRAIN_RATIO
    )
    
    # 4. Compute baselines
    baselines = []
    baselines.append(majority_class_baseline(y_train, y_test))
    baselines.append(odds_baseline(X_train, X_test, y_train, y_test))
    
    # 5. Train main model
    model, scaler, metrics = train_logistic_regression(
        X_train, X_test, y_train, y_test, feature_cols
    )
    
    # 6. Save artifacts
    results = save_artifacts(
        model, scaler, metrics, baselines, 
        split_info, feature_cols, project_root
    )
    
    # 7. Print summary
    print_summary(results)
    
    print("\n" + "=" * 60)
    print("[DONE] Baseline training complete!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
