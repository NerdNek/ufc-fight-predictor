# FightSense AI - UFC Fight Outcome Predictor

A machine learning project that predicts UFC fight outcomes using pre-fight statistics, historical data, and betting odds. Supports both historical backtesting and hypothetical/upcoming matchup prediction.

## Project Structure

```
ufc-fight-predictor/
├── data/
│   ├── raw/               # Original datasets (not tracked in git)
│   └── processed/         # Cleaned, leakage-safe datasets
├── models/                # Trained model artifacts
├── notebooks/             # Jupyter notebooks for EDA and experiments
├── reports/               # Generated analysis and metrics
├── src/
│   ├── clean.py           # Data cleaning (leakage removal)
│   ├── features.py        # Feature engineering (differentials)
│   ├── predict.py         # Dual-mode inference engine
│   ├── train_baseline.py  # Baseline model training
│   ├── train_no_odds.py   # No-odds HGB model training
│   └── train_tree_day4a.py # Tree model with interaction features
└── streamlit_app.py       # Web UI for predictions
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/NerdNek/ufc-fight-predictor.git
cd ufc-fight-predictor
pip install -r requirements.txt

# 2. Add data
# Place ufc-master.csv in data/raw/

# 3. Run pipeline
python src/clean.py             # Remove leakage columns
python src/features.py          # Generate differential features
python src/train_tree_day4a.py  # Train with-odds HGB model
python src/train_no_odds.py     # Train no-odds HGB model

# 4. Run the web UI
streamlit run streamlit_app.py

# Or predict from the command line
python src/predict.py --red "Israel Adesanya" --blue "Sean Strickland"
```

## Prediction Modes

The predictor operates in two modes, selected automatically based on whether the fighter pair exists in the dataset.

### Mode A: Historical (with-odds model)

When the exact matchup exists in `features_with_odds.csv`, the system uses the real fight-level row with actual betting odds. This gives the highest-fidelity backtest since the model was trained with odds features and odds-gated interaction features.

- Uses `hgb_day4a.joblib` (86 features, ROC-AUC 0.692)
- Includes odds-based interaction features (close-odds gates, favorite/underdog gates)
- Best for validating model performance on known fights

### Mode B: Synthetic (no-odds model)

When the matchup does not exist, the system constructs a synthetic feature row from each fighter's most recent individual stats. Because betting lines are unavailable for hypothetical fights, this mode uses a dedicated no-odds model.

- Uses `hgb_no_odds.joblib` (68 features, ROC-AUC 0.600)
- Builds differential features on-the-fly from per-fighter career stats
- Supports optional fight context: weight class, title bout, number of rounds
- Best for upcoming or hypothetical matchups

## Model Performance

### Evaluation Methodology
- **Split**: Time-based (no shuffle) - train on past, test on future
- **Train**: 5,222 fights (2010-03-21 to 2022-05-21)
- **Test**: 1,306 fights (2022-05-21 to 2024-12-07)

### Results

| Model | Accuracy | ROC-AUC | Log Loss |
|-------|----------|---------|----------|
| Majority Class (always Red) | 56.28% | - | - |
| **Odds Only (LogReg)** | **66.85%** | **0.723** | - |
| Logistic Regression (77 features) | 62.94% | 0.699 | - |
| LogReg v2 (rate-based diffs, 73 features) | 63.55% | 0.699 | - |
| HGB Odds-Only | 63.25% | 0.678 | 0.64 |
| HGB Skill-Only | 59.19% | 0.598 | 0.68 |
| HGB Full (86 features, with odds) | 63.17% | 0.692 | 0.64 |
| HGB No-Odds (68 features) | 58.42% | 0.600 | 0.69 |

### Key Findings

- **Odds-only remains strongest overall** (AUC 0.723), consistent with market efficiency
- **Ablation study confirms**: odds features alone carry most predictive signal (AUC 0.678); skills add +0.014 incremental AUC
- **Segment analysis**: Model shows +8.3% lift over majority baseline in confident-market fights
- **No-odds model**: Lower accuracy as expected, but enables prediction for any fighter pairing without requiring betting lines

## Data Pipeline

### 1. Leakage Prevention (clean.py)
Removes post-fight columns: `Finish`, `FinishRound`, `TotalFightTimeSecs`, etc.

### 2. Feature Engineering (features.py)
- 77 differential features (Red - Blue)
- Removes fighter identity, forces matchup reasoning
- Encodes stance matchups, weight classes, rankings
- Produces with-odds and no-odds feature variants

### 3. Model Training
- **train_tree_day4a.py**: With-odds HGB model with interaction features, segment evaluation, ablation study
- **train_no_odds.py**: No-odds HGB model for synthetic matchup prediction
- **train_baseline.py**: Logistic regression baseline

### 4. Inference (predict.py)
- Dual-mode: historical lookup (Mode A) or synthetic construction (Mode B)
- Fighter profile extraction from most recent appearance in dataset
- Schema lock enforcement to prevent feature drift

## Streamlit Web UI

The web interface provides:
- Fighter selection from all fighters in the dataset
- Optional fight context controls (weight class, title bout, number of rounds)
- Automatic mode selection with clear labeling
- Probability display for both corners

## License

MIT
