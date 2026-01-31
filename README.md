# FightSense AI - UFC Fight Outcome Predictor

A machine learning project to predict UFC fight outcomes using pre-fight statistics and historical data.

## Project Structure

```
ufc-fight-predictor/
├── data/
│   ├── raw/           # Original datasets (not tracked in git)
│   └── processed/     # Cleaned, leakage-safe datasets
├── notebooks/         # Jupyter notebooks for EDA and experiments
├── reports/           # Generated analysis and figures
└── src/               # Source code
    └── clean.py       # Data cleaning script
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ufc-fight-predictor.git
cd ufc-fight-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your data:
   - Place `ufc-master.csv` in `data/raw/`

4. Run data cleaning:
```bash
python src/clean.py
```

## Data Leakage Prevention

The `clean.py` script removes post-fight information to prevent data leakage:

| Removed Column | Reason |
|----------------|--------|
| `Finish` | Fight outcome (KO, SUB, DEC) |
| `FinishDetails` | Specific finish technique |
| `FinishRound` | Round fight ended |
| `FinishRoundTime` | Time in round |
| `TotalFightTimeSecs` | Total fight duration |

The `Winner` column is preserved as the **target variable** for prediction.

## Dataset

After cleaning:
- **6,528 fights**
- **113 features** (pre-fight statistics)
- **Target**: Winner (Red: 58%, Blue: 42%)

## License

MIT
