# FightSense AI - UFC Fight Outcome Predictor

A machine learning project to predict UFC fight outcomes using pre-fight statistics and historical data.

## Project Structure

```
ufc-fight-predictor/
├── data/
│   ├── raw/           # Original datasets (not tracked in git)
│   └── processed/     # Cleaned, leakage-safe datasets
├── models/            # Trained model artifacts
├── notebooks/         # Jupyter notebooks for EDA and experiments
├── reports/           # Generated analysis and metrics
└── src/
    ├── clean.py           # Data cleaning (leakage removal)
    ├── features.py        # Feature engineering (differentials)
    ├── train_baseline.py  # Baseline model training
    └── train_tree_day4a.py # Tree model with interaction features
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
python src/clean.py           # Remove leakage columns
python src/features.py        # Generate differential features
python src/train_baseline.py  # Train baseline model
python src/train_tree_day4a.py # Train tree model with ablation
```

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
| HGB Odds-Only | 63.25% | 0.678 | 0.64 |
| HGB Skill-Only | 59.19% | 0.598 | 0.68 |
| HGB Full (89 features) | 63.17% | 0.692 | 0.64 |

### Key Findings

- **Odds-only remains strongest overall** (AUC 0.723), consistent with market efficiency
- **Ablation study confirms**: odds features alone carry most predictive signal (AUC 0.678); skills add +0.014 incremental AUC
- **Segment analysis**: Model shows +8.3% lift over majority baseline in "confident market" fights (not-close-odds segment)
- **Further work**: Focus on calibration, feature selection, and alternative model architectures

### Ablation Study Insight

The tree model with gated interactions approached odds performance in confident-market fights. Skill features provide modest incremental value when combined with odds, but do not outperform the market on their own.

## Data Pipeline

### 1. Leakage Prevention (clean.py)
Removes post-fight columns: `Finish`, `FinishRound`, `TotalFightTimeSecs`, etc.

### 2. Feature Engineering (features.py)
- **77 differential features** (Red - Blue)
- Removes fighter identity, forces matchup reasoning
- Encodes stance matchups, weight classes, rankings

### 3. Baseline Training (train_baseline.py)
- Time-based train/test split
- Standardized features
- Logistic regression with balanced class weights

### 4. Tree Model (train_tree_day4a.py)
- Interaction features (close-odds gates, favorite/underdog gates)
- HistGradientBoostingClassifier with segment evaluation
- Ablation study comparing odds-only, skill-only, and full models

## License

MIT
