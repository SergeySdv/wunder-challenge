# Feature Generation Lab – catch22 & Shared Features

This guide explains how to use automated time-series feature generation (via **catch22**) together with our lag-based MLP and CatBoost experiments.

The key idea:  
**Compute rich features offline once, save them as a small table, and then reuse them in both MLP and CatBoost without adding heavy dependencies to `solution.py`.**

---

## 1. What This Is For

- Provide additional, higher-level time-series descriptors beyond simple lags/deltas/rolling stats.
- Keep all heavy work **offline**:
  - `compute_catch22_features.py` computes per-sequence, per-dimension catch22 features and saves them to `datasets/catch22_per_seq.npz`.
- Allow **offline labs** to use the enriched feature set:
  - The lag-MLP in `train_model.py` (v4 experiments), and
  - The offline CatBoost lab (`train_catboost_experiment.py`, `catch22_feature_importance.py`),
  can both consume v3 features + per-sequence catch22.

Important: **per-sequence catch22 vectors are not safe for direct use in the final submission model**, because they encode full-sequence information and sequence identity from `train.parquet`. They are therefore reserved for offline analysis and for **designing streaming-safe analogs** that can be computed on-the-fly from short lag windows.

---

## 2. Offline Feature Extraction with catch22

### 2.1. Dependencies

Install the `catch22` Python package inside your `.venv`:

```bash
cd /Users/sergei/PycharmProjects/WunderSex
source .venv/bin/activate
pip install catch22
```

### 2.2. Script: `compute_catch22_features.py`

The script `compute_catch22_features.py`:

- Loads the competition data:
  - `datasets/train.parquet`, with columns:
    - `seq_ix`, `step_in_seq`, `need_prediction`, features `0`..`31`.
- For each `seq_ix`:
  - For each feature column `0`..`31`:
    - Extracts the full 1000-length series as a 1D numpy array,
    - Computes `catch22_all(series)` to get 22 canonical features.
- Accumulates results into:
  - `catch22_values`: shape `(n_seqs, n_dims, 22)` (float32),
  - `seq_ids`: shape `(n_seqs,)` (int),
  - `feature_cols`: shape `(n_dims,)` (str),
  - `catch22_names`: shape `(22,)` (str).
- Saves to:

```bash
datasets/catch22_per_seq.npz
```

You can run it as:

```bash
cd /Users/sergei/PycharmProjects/WunderSex/competition_package
source ../.venv/bin/activate
python compute_catch22_features.py
```

This is a **one-time or occasional** operation; it may take from minutes up to ~1 hour depending on CPU.

---

## 3. Using catch22 Features in `train_model.py` / CatBoost Labs

Once `datasets/catch22_per_seq.npz` exists:

- `train_model.py`:
  - Loads the `.npz` once at the top level and builds a mapping `seq_ix -> 704-dim vector` by flattening:

    ```python
    # catch22_values: (n_seqs, n_dims, 22)
    # feature_cols: ["0", ..., "31"]
    _catch22_flat = catch22_values.reshape(n_seqs, n_dims * 22)
    seq_to_catch22 = {int(seq_ix): _catch22_flat[i] for i, seq_ix in enumerate(seq_ids)}
    ```

  - Inside `build_supervised_dataset`, when `use_catch22=True` and constructing `X_t`:
    - After `[lag_flat, delta_flat, mean_last10, std_last10, step_feature]`,
    - Looks up the per-sequence catch22 vector for `seq_ix` and appends it.

  - Result: with `n_lags=10`, each sample has:

    ```text
    X_t = [lag_flat, delta_flat, mean_last10, std_last10, step_feature, catch22_seq_features]
    ```

    giving a **1409-dim** feature vector (705 v3 dims + 704 catch22 dims).

  - The MLP trained with `use_catch22=True` (v4) is used only for lab runs to measure offline R² and get a sense of how much signal the per-sequence catch22 block adds.

- `train_catboost_experiment.py` and `catch22_feature_importance.py`:
  - Reuse `build_supervised_dataset(..., use_catch22=True)` to get the same v4 feature set.
  - Train CatBoost MultiRMSE models to:
    - Benchmark performance on the enriched feature set, and
    - Compute feature importances for the catch22 block, aggregated by statistic and by dimension.
  - Early analysis (see `EXPERIMENT_LOG.md` entry 2.8) shows:
    - Base v3 features still carry ~88% of total CatBoost importance.
    - The catch22 block contributes ~12%, dominated by:
      - Spectral stats (Welch band power and spectral centroid),
      - Autocorrelation/time-scale stats (e.g. `CO_f1ecac`, `CO_FirstMin_ac`),
      - Simple persistence / time-reversibility descriptors (`CO_trev_1_num`, `SB_BinaryStats_mean_longstretch1`),
      - Local trend/residual ratios.

These results guide what **streaming-friendly analogs** we should prototype next (e.g., short-window autocorrelation, simple energy/variance proxies, persistence indicators).

---

## 4. Streaming-Safe Analogs (Future Work)

Because per-sequence catch22 vectors leak full-sequence information and do not transfer to new hidden sequences, the plan is to:

1. Use catch22 **offline only** (as above) to discover which types of statistics matter most (spectral, autocorrelation, persistence, etc.).  
2. Design cheap, streaming-safe approximations that can be computed from the last `n_lags` steps only, per feature, such as:
   - Short-window variance / energy and absolute deviation,
   - Simple lag-1 / lag-2 autocorrelation estimates,
   - Rolling skewness/kurtosis,
   - Simple persistence indicators (e.g., fraction of steps above a local mean).  
3. Wire these analogs into `train_model.py` (as additional features after the existing v3 block), retrain the MLP, and evaluate offline/streaming R².  
4. Mirror only these streaming-safe analogs in `solution.py` for submission, keeping the heavy catch22 machinery strictly offline.  

This way, we get the **conceptual benefits** of catch22 (it tells us which dynamics matter) without leaking full sequence identity into the submission model. 

---

## 5. Recommended Workflow

1. **Generate catch22 features once**:
   - Run `compute_catch22_features.py` after installing `pycatch22` or a compatible implementation.
2. **Use catch22 in offline labs**:
   - Train MLPs and CatBoost models with `use_catch22=True` to understand which catch22 statistics and which state dimensions are most informative.
   - Optionally cluster sequences in catch22 space to identify different “types” of dynamics.
3. **Design and test streaming-safe analogs**:
   - Add small, cheap statistics inspired by the important catch22 patterns to `train_model.py`.
   - Retrain and compare validation / streaming R² to the v3 baseline.
4. **Keep submission clean**:
   - Only include features in `solution.py` that are computable from the rolling `n_lags` buffer and current `step_in_seq`.
   - Do not use per-sequence catch22 vectors directly in the submission model.

This keeps the submission model simple and leak-free while still leveraging powerful automated feature generation tools offline to guide feature engineering. 
