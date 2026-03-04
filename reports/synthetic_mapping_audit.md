# Synthetic Mapping Audit — Pre-Existing Dif Columns

## Summary

Three pre-existing differential columns in the training data (`SigStrDif`, `AvgSubAttDif`, `AvgTDDif`) are **NOT** computed from the same raw columns that `build_synthetic_row()` in `predict.py` uses. This causes a **scale mismatch** in Mode B (synthetic) predictions.

---

## Column-by-Column Audit

### `SigStrDif`

| | Training (pre-existing) | Synthetic builder |
|---|---|---|
| **Source** | Pre-computed in `ufc-master.csv` (unknown origin, no corresponding Red/Blue columns in dataset) | `RedAvgSigStrLanded - BlueAvgSigStrLanded` (per-minute rate) |
| **Scale** | Range: [-118, +128], std=19.58 | Typical range: ~[-10, +10] |
| **Mismatch** | 6,079 / 6,528 rows differ by >0.1 | Max error: 256.44 |

**Root cause**: `SigStrDif` in the raw dataset is a **black-box pre-computed column**. It was NOT derived from `RedAvgSigStrLanded` / `BlueAvgSigStrLanded`. The original source columns (likely career cumulative significant strikes) are not present in the CSV.

The synthetic builder (`predict.py` line 416) maps:
```python
"SigStrDif": ("AvgSigStrLanded", "AvgSigStrLanded"),
```
This uses per-minute *rate* stats, but the training column uses a **different (likely cumulative) scale**.

### `AvgSubAttDif`

| | Training (pre-existing) | Synthetic builder |
|---|---|---|
| **Source** | Pre-computed in `ufc-master.csv` | `RedAvgSubAtt - BlueAvgSubAtt` |
| **Mismatch** | 4,971 / 6,528 rows differ by >0.1 | Max error: 16.80 |

### `AvgTDDif`

| | Training (pre-existing) | Synthetic builder |
|---|---|---|
| **Source** | Pre-computed in `ufc-master.csv` | `RedAvgTDLanded - BlueAvgTDLanded` |
| **Mismatch** | 5,716 / 6,528 rows differ by >0.1 | Max error: 22.00 |

---

## Code Locations

### `predict.py` — `build_synthetic_row()` (line 403-418)

```python
stat_to_generic = {
    ...
    "SigStrDif": ("AvgSigStrLanded", "AvgSigStrLanded"),    # WRONG
    "AvgSubAttDif": ("AvgSubAtt", "AvgSubAtt"),             # WRONG
    "AvgTDDif": ("AvgTDLanded", "AvgTDLanded"),             # WRONG
    ...
}
```

### `predict.py` — `EXISTING_DIFF_PAIRS` (line 55-71)

```python
EXISTING_DIFF_PAIRS = {
    ...
    "SigStrDif": ("RedAvgSigStrLanded", "BlueAvgSigStrLanded"),    # Also WRONG
    "AvgSubAttDif": ("RedAvgSubAtt", "BlueAvgSubAtt"),             # Also WRONG
    "AvgTDDif": ("RedAvgTDLanded", "BlueAvgTDLanded"),             # Also WRONG
    ...
}
```

### `features.py` — `EXISTING_DIFF_COLS` (line 79-84)

```python
EXISTING_DIFF_COLS = [
    ...
    'SigStrDif', 'AvgSubAttDif', 'AvgTDDif'   # Used AS-IS from raw data ✓
]
```

The training pipeline correctly uses these columns **as-is** — it never recomputes them. The bug is exclusively in the synthetic builder (Mode B inference path).

---

## Conclusion

> **`SigStrDif` is computed from `AvgSigStrLanded` (per-minute rate), which is wrong because training uses a pre-computed Dif column from the raw dataset that operates on a different (cumulative) scale.**

The same applies to `AvgSubAttDif` and `AvgTDDif`.

Since the original source columns for these Dif values are not present in the dataset, the fix options are:
1. **Drop** `SigStrDif`, `AvgSubAttDif`, `AvgTDDif` from the model entirely (and retrain)
2. **Recompute** these Dif columns in the training pipeline from `RedAvgSigStrLanded - BlueAvgSigStrLanded` (consistent with what the synthetic builder can produce), then retrain
3. **Accept the mismatch** and add z-score clamping as a safety valve
