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
    └── train_baseline.py  # Baseline model training
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
```

## Model Performance

### Evaluation Methodology
- **Split**: Time-based (no shuffle) - train on past, test on future
- **Train**: 5,222 fights (2010-03-21 to 2022-05-21)
- **Test**: 1,306 fights (2022-05-21 to 2024-12-07)

### Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Majority Class (always Red) | 56.28% | - |
| Odds Only (single feature) | 66.85% | 0.723 |
| **Logistic Regression (77 features)** | **62.94%** | **0.699** |

### Key Findings
- ✅ Beat majority baseline (+6.7%)
- ✅ ROC-AUC > 0.60 milestone
- ⚠️ Odds-only baseline is very strong (markets are efficient)
- 📈 Room for improvement with advanced models

### Top Predictive Features
1. `ev_diff` - Expected value difference
2. `odds_diff` - Betting odds difference
3. `ReachDif` - Reach advantage
4. `dec_split_wins_diff` - Decision experience

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

## License

MIT
