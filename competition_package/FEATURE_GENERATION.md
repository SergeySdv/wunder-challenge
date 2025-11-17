# Feature Generation Lab – catch22 & Shared Features

This guide explains how to use automated time-series feature generation (via **catch22**) together with our lag-based MLP and CatBoost experiments.

The key idea:  
**Compute rich features offline once, save them as a small table, and then reuse them in both MLP and CatBoost without adding heavy dependencies to `solution.py`.**

---

## 1. What This Is For

- Provide additional, higher-level time-series descriptors beyond simple lags/deltas/rolling stats.
- Keep all heavy work **offline**:
  - `compute_catch22_features.py` (to be implemented) will:
    - Load `datasets/train.parquet`,
    - Compute per-sequence, per-dimension catch22 features,
    - Save them to `datasets/catch22_per_seq.npz`.
- Allow **both**:
  - The lag-MLP (in `train_model.py` / `solution.py`), and
  - The offline CatBoost lab (`train_catboost_experiment.py`),
  to use the **same enriched feature set**.

We do *not* call catch22 in `solution.py` at runtime; we only read a small `.npz` file produced offline.

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

The planned script (see `compute_catch22_features.py`) will:

- Load the competition data:
  - `datasets/train.parquet`, with columns:
    - `seq_ix`, `step_in_seq`, `need_prediction`, features `0`..`31`.
- For each `seq_ix`:
  - For each feature column `0`..`31`:
    - Extract the full 1000-length series as a 1D numpy array,
    - Compute `catch22_all(series)` to get 22 canonical features.
- Accumulate results into:
  - `catch22_values`: shape `(n_seqs, n_dims, 22)` (float32),
  - `seq_ids`: shape `(n_seqs,)` (int),
  - `feature_cols`: shape `(n_dims,)` (str),
  - `catch22_names`: shape `(22,)` (str).
- Save to:

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

## 3. Using catch22 Features in `train_model.py` (MLP)

Once `datasets/catch22_per_seq.npz` exists:

- Modify `train_model.py` (feature builder) to:
  - Load the `.npz` once at the top-level,
  - Build a mapping `seq_ix -> 704-dim vector` by flattening:

    ```python
    # catch22_values: (n_seqs, n_dims, 22)
    # feature_cols: ["0", ..., "31"]
    # flatten along (dim, feature) for each seq_ix
    flattened = catch22_values.reshape(n_seqs, n_dims * 22)
    ```

  - Inside `build_supervised_dataset`, when constructing `X_t`:
    - After `[lag_flat, delta_flat, mean_last10, std_last10, step_feature]`,
    - Look up the per-sequence catch22 vector for `seq_ix` and append it.

Result:

- New feature vector per sample:

  ```text
  [lag_flat, delta_flat, mean_last10, std_last10, step_feature, catch22_seq_features]
  ```

- Train the MLP as usual and compare validation mean R² to the current v3 baseline (~0.428).

This lets the MLP act as a **fast feature lab** for evaluating whether catch22 features actually help.

---

## 4. Using catch22 Features in `solution.py` (Streaming MLP)

To use the same features in the submission model:

- Bundle `datasets/catch22_per_seq.npz` into the submission zip.
- In `PredictionModel.__init__`:
  - Load `.npz` and construct the same `seq_ix -> flattened catch22 vector` mapping.
- In `_build_features`:
  - Build `[lag_flat, delta_flat, mean_last10, std_last10, step_feature]` as now,
  - Append the corresponding per-sequence catch22 vector.
- Normalize with the updated `x_mean`, `x_std` from `models/lag_mlp_normalization.npz`.

No `import catch22` is needed inside `solution.py`; it remains a simple, dependency-light file.

---

## 5. Using catch22 Features in `train_catboost_experiment.py`

Because CatBoost experiment script already uses `build_supervised_dataset`:

- After integrating catch22 features into `train_model.py`:
  - `train_catboost_experiment.py` automatically benefits from the same enriched `X` when you import and call `build_supervised_dataset`.
- This means:
  - **MLP and CatBoost share exactly the same feature set**,
  - You can:
    - Quickly test new features with MLP,
    - Occasionally run a heavy CatBoost MultiRMSE training as a “strong oracle” on the best feature set.

---

## 6. Recommended Workflow

1. **Generate catch22 features once**:
   - Run `compute_catch22_features.py` after installing `catch22`.
2. **Wire them into `train_model.py`**:
   - Append per-sequence catch22 vectors to `X_t`.
   - Retrain the MLP and record validation mean R².
3. **Update `solution.py`**:
   - Append the same catch22 vectors in `_build_features`.
   - Check streaming R² and runtime with `python solution.py`.
4. **Log results**:
   - Record configs and scores in `EXPERIMENT_LOG.md`.
5. **Optional CatBoost run**:
   - Run `train_catboost_experiment.py` on the enriched features if you want an upper-bound check.

This keeps the submission model simple and fast, while leveraging powerful automated feature generation offline for both MLP and CatBoost. 

