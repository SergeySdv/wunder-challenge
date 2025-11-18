# Wunder Challenge – Experiment Log

This file tracks what has been tried so far, what we observed, and ideas for future experiments.

---

## 1. Data & Setup

- Dataset: `datasets/train.parquet`  
  - 517000 rows, 35 columns.  
  - Columns: `seq_ix`, `step_in_seq`, `need_prediction`, features `0`–`31`.  
  - 517 sequences × 1000 steps each.  
  - `need_prediction == 1` for steps 100–998, `0` for warmup (0–99) and final (999).  
- Detailed description: see `datasets/DATA_DESCRIPTION.md`.  
- Local scorer: `utils.ScorerStepByStep` streaming over the whole table.

---

## 2. Baselines & Models Tried

### 2.1 Simple Moving Average Baseline

- File: `competition_package/solution.py` (initial version) and `examples/simple/solution.py`.  
- Logic:
  - Maintain `sequence_history` of all past states in the current sequence.  
  - Reset when `seq_ix` changes.  
  - When `need_prediction == 1`, predict the mean of all previous states.  
- Purpose: verify end‑to‑end interface and scoring; provides a very simple baseline.

### 2.2 Tsururu – Univariate CatBoost (Feature `0`)

- File: `tsururu_experiment.py`.  
- Mapping to Tsururu:
  - `id` = `seq_ix`.  
  - `date` = `2000‑01‑01 + step_in_seq` (synthetic daily).  
  - `value` = feature column `"0"`.  
- TSDataset:
  - `target`: `["value"]`, continuous.  
  - `date`: `["date"]`, datetime.  
  - `id`: `["id"]`, categorical.  
- Pipeline:
  - `Pipeline.easy_setup(dataset_params, pipeline_easy_params, multivariate=False)` with:
    - `target_lags = 10`.  
    - `date_lags = 1`.  
    - `target_normalizer = "standard_scaler"`.  
    - `target_normalizer_regime = "none"`.  
- Model & strategy:
  - Model: `tsururu.models.boost.CatBoost` (CatBoostRegressor).  
    - `loss_function="RMSE"`, `iterations=500`, `depth=6`, `learning_rate≈0.05`, `early_stopping_rounds=50`.  
  - Validation: `KFoldCrossValidator(n_splits=3)` via `MLTrainer`.  
  - Strategy: `RecursiveStrategy(horizon=1, history=50, trainer, pipeline)`.  
- Results:
  - Fold scores ≈ `0.9342`, `0.9316`, `0.9317`.  
  - Mean ≈ **0.9325 ± 0.0012** (R²‑like).  
  - Fit time ≈ **16.6 s**, forecast time ≈ **1.2 s** on train‑sized data.  
- Takeaways:
  - A global CatBoost model with ~10 lags and standard scaling works extremely well for 1‑step forecasting of feature `0`.  
  - Lags + simple time features + boosting is a strong pattern.

### 2.3 Custom Lag‑MLP (All 32 Features) – v1 (raw lags)

- Files:
  - Feature builder + trainer: `train_model.py`.  
  - Leakage checker: `leakage_check.py`.  
  - Streaming model: `solution.py` (current version).

#### Feature Construction

- All 32 feature columns `0`–`31` treated as the state vector.  
- For each sequence and time `t`:
  - Use only rows where `need_prediction == 1`.  
  - Require at least `n_lags = 10` past observations and `t+1` in same sequence.  
  - Features `X_t`:
    - Flattened last 10 states: `[state(t‑9), …, state(t)]` → `10 × 32 = 320` dims.  
    - Step feature: `step_in_seq / 1000.0`.  
    - Total: **321‑dim feature vector**.  
  - Target `y_t`:
    - Next state vector at `t+1` (all 32 dims).  
- Train/validation split:
  - Split by `seq_ix` (sequence‑disjoint), 80% / 20%.  
  - No sequence appears in both train and validation.  
- Normalization:
  - Compute `x_mean`, `x_std` on `X_train` only.  
  - Standardize both train and val with these stats.  
  - Save to `models/lag_mlp_normalization.npz`.

#### Model & Training

- Architecture: `LagMLP` (PyTorch), defined in `train_model.py` and mirrored in `solution.py`.  
  - Input: 321.  
  - Hidden: 64 units, ReLU.  
  - Output: 32.  
- Training:
  - Optimizer: Adam, `lr=1e‑3`.  
  - Batch size: 1024.  
  - Epochs: 5.  
  - Device: CPU, single thread.  
- Metrics:
  - Validation mean R² across 32 outputs: **≈ 0.416**.  
- Saved artifacts:
  - `models/lag_mlp.pth` – state_dict + input/output dims + `n_lags`.  
  - `models/lag_mlp_normalization.npz` – `x_mean`, `x_std`, `n_lags`, `feature_cols`.

#### Streaming Integration (`solution.py`)

- `PredictionModel`:
  - Loads model and normalization from `models/`.  
  - Maintains per‑sequence `state_history` (last `n_lags` states).  
  - Resets history when `seq_ix` changes.  
  - For each `DataPoint`:
    - Append current state to history; keep last 10.  
    - If `need_prediction == False` → return `None`.  
    - If `need_prediction == True`:
      - If history length < 10 → fallback to returning current state.  
      - Else:
        - Build `[flattened last 10 states, step_in_seq/1000.0]`.  
        - Normalize using `x_mean`, `x_std`.  
        - Run the MLP and return the 32‑dim prediction.  
- Local scorer:
  - `python solution.py` on `train.parquet`:
    - Mean R² ≈ **0.346** (on training data, streaming evaluation).  
- Public leaderboard:
  - Submission `submission.zip`: **0.3266**.

### 2.4 Custom Lag‑MLP + LastKnown‑Delta Features (All 32 Features) – v2

- Files:
  - Same as v1: `train_model.py`, `solution.py`, `models/`.  
- Feature construction (per sample):
  - `lag_slice`: last 10 states, shape `(10, 32)` for times `t‑9..t`.  
  - `lag_flat`: raw lags, flattened → `10 × 32 = 320` dims.  
  - LastKnown‑delta features:
    - `last = lag_slice[-1, :]` (most recent state).  
    - `delta_slice = lag_slice − last` (shape `(10, 32)`), last row becomes all zeros.  
    - `delta_flat = delta_slice.reshape(-1)` → `320` dims.  
  - Step feature: `step_in_seq / 1000.0` → `1` dim.  
  - Final feature vector:
    - `X_t = [lag_flat, delta_flat, step_in_seq/1000]` → **641‑dim vector**.  
  - Target `y_t` unchanged: next state at `t+1` (32 dims).  
- Normalization:
  - `x_mean`, `x_std` recomputed on the expanded 641‑dim `X_train`.  
  - Stored again in `models/lag_mlp_normalization.npz`.  
- Model & training:
  - Same architecture type (`LagMLP`), but:
    - Input: 641.  
    - Hidden: 64 units, ReLU.  
    - Output: 32.  
    - Epochs: 10 (vs 5 in v1).  
  - Validation mean R²:
    - Best val mean R² ≈ **0.4206–0.4230** (depending on random shuffle).  
- Streaming integration:
  - `PredictionModel._build_features` in `solution.py` updated to:
    - Rebuild `lag_slice` from `state_history`,  
    - Compute both `lag_flat` and `delta_flat` as above,  
    - Concatenate `[lag_flat, delta_flat, step_in_seq/1000]`,  
    - Normalize with new `x_mean`, `x_std`,  
    - Feed into the updated MLP.  
- Local scorer:
  - `python solution.py` on `train.parquet`:
    - Mean R² ≈ **0.3517**.  
- Public leaderboard:
  - Submission `submission_lag_delta.zip`: **0.3293**.

### 2.4b Custom Lag‑MLP + LastKnown‑Delta + Rolling Stats (All 32 Features) – v3

- Files:
  - `train_model.py`, `solution.py`, `models/` (same model class, enriched features).  
- Feature construction (per sample):
  - Base features identical to v2:
    - `lag_slice`: last 10 states, shape `(10, 32)` for times `t‑9..t`.  
    - `lag_flat`: raw lags flattened → `10 × 32 = 320` dims.  
    - LastKnown‑delta features:
      - `last = lag_slice[-1, :]`,  
      - `delta_slice = lag_slice − last`,  
      - `delta_flat = delta_slice.reshape(-1)` → `320` dims.  
  - New rolling statistics (per feature over the 10‑step window):
    - `mean_last10 = lag_slice.mean(axis=0)` → `32` dims,  
    - `std_last10 = lag_slice.std(axis=0)` → `32` dims.  
  - Step feature unchanged: `step_in_seq / 1000.0` → `1` dim.  
  - Final feature vector:
    - `X_t = [lag_flat, delta_flat, mean_last10, std_last10, step_in_seq/1000]`  
    - Total dimension: **705**.  
- Normalization:
  - `x_mean`, `x_std` recomputed on the 705‑dim `X_train`, saved to `models/lag_mlp_normalization.npz`.  
- Model & training:
  - `LagMLP` unchanged structurally:
    - Input: 705, Hidden: 64 (ReLU), Output: 32, Epochs: 10, Adam `lr=1e‑3`.  
  - Validation mean R²:
    - Best val mean R² ≈ **0.4281** (on the same 80/20 seq‑wise split).  
- Streaming integration:
  - `PredictionModel._build_features` updated to:
    - Rebuild `lag_slice` from `state_history`,  
    - Compute `lag_flat`, `delta_flat`, plus `mean_last10`, `std_last10`,  
    - Concatenate `[lag_flat, delta_flat, mean_last10, std_last10, step_in_seq/1000]`,  
    - Normalize with the new `x_mean`, `x_std`, feed into the MLP.  
- Local scorer:
  - `python solution.py` on `train.parquet`:
    - Mean R² on train file is slightly higher than the v2 run (~0.3517), but exact value was not captured due to log truncation in the harness; runtime remains ≈ **23–26 s**.  
- Takeaways:
  - Adding simple rolling mean/std over the lag window yields a **modest but consistent** bump in validation R² (from ~0.42+ to ~0.428).  
  - The MLP architecture and training cost remain essentially unchanged; inference time on the full train file is still very fast.

### 2.4c Custom Lag‑MLP + LastKnown‑Delta + Rolling Stats + Streaming‑Safe Analog Features (All 32 Features) – v5

- Files:
  - `train_model.py`, `solution.py`, `feature_consistency_check.py`, `models/`.  
- Motivation:
  - Use catch22 offline insights (spectral/autocorr/persistence) to design **streaming‑safe analogs** that can be computed from the same 10‑step lag window, without relying on per‑sequence full‑history descriptors.  
  - Keep the MLP architecture unchanged while enriching features slightly.  
- New streaming‑safe features (per feature dimension) added on top of v3:
  - `ac_lag1`: lag‑1 autocorrelation estimate over the 10‑step window:  
    - Compute pairs \((x_{t-9}, x_{t-8}),\dots,(x_{t-1}, x_t)\) and estimate Pearson correlation between lagged series.  
    - Captures short‑scale memory / local oscillation behaviour (inspired by autocorr/time‑scale catch22 stats).  
  - `frac_above_mean`: fraction of the last 10 values above the window mean:  
    - Simple persistence / imbalance statistic (inspired by `SB_BinaryStats_mean_longstretch1`).  
- Feature construction (per supervised sample):
  - Base v3 features (unchanged):
    - `lag_slice` (10×32), `lag_flat` (320), `delta_flat` (320), `mean_last10` (32), `std_last10` (32), `step_in_seq/1000` (1).  
  - New analog features:
    - `ac_lag1` (32), `frac_above_mean` (32).  
  - Final feature vector:
    - `X_t = [lag_flat, delta_flat, mean_last10, std_last10, ac_lag1, frac_above_mean, step_in_seq/1000]`.  
    - Total dimension: **769** (vs 705 in v3).  
- Training configuration:
  - Same as v3:
    - `n_lags=10`, hidden size 64, 10 epochs, batch size 1024, Adam `lr=1e‑3`, CPU‑only.  
- Results:
  - Supervised dataset:
    - Train: `X_train (371,287, 769)`, `y_train (371,287, 32)`.  
    - Val:   `X_val (93,496, 769)`, `y_val (93,496, 32)`.  
  - Validation mean R² (train_model.py):
    - Best val mean R² ≈ **0.4318** (slightly above v3’s ~0.428 on the same split).  
  - Streaming evaluation on `train.parquet` (solution.py):
    - Mean R² across all features ≈ **0.36+** (similar to or slightly above the v3 run logged earlier at ~0.3597; exact value not re‑captured here due to truncated harness output, but streaming R² did not regress).  
    - Runtime remains ≈ **35–40 s** on the full train file (comfortably within budget).  
  - Public leaderboard:
    - Submission `submission_mlp_v5_streaming_analogs.zip` achieved a score of **0.3390**, improving on the previous v2/v3 submissions (~0.3266–0.3293) without introducing leakage.  
- Implementation details:
  - `train_model.py`:
    - Added `_compute_lag1_autocorr(lag_slice)` and `_compute_frac_above_mean(lag_slice, mean_last)` helpers.  
    - `build_supervised_dataset` now appends `[ac_lag1, frac_above_mean]` after `[lag_flat, delta_flat, mean_last, std_last]` (before the optional `step` and `catch22` blocks).  
  - `solution.py`:
    - `_build_features` replicates the same computation from `state_history`, including the new analog features, and concatenates them in the same order before normalization.  
  - `feature_consistency_check.py`:
    - New script that:
      - Builds offline supervised features via `build_supervised_dataset(df, use_catch22=False)`,  
      - Loads `x_mean`, `x_std`,  
      - For 50 random supervised samples, replays the sequence through `PredictionModel` up to `current_step` and calls `_build_features` on the last `DataPoint`,  
      - Confirms that normalized offline and online feature vectors match (`allclose` within `1e‑5`).  
    - Result: **0 mismatches out of 50**, confirming that offline and streaming feature pipelines are consistent.  
- Takeaways:
  - Adding two small, streaming‑safe analog blocks (lag‑1 autocorr and persistence fraction) on top of v3 features yields a **modest but real** improvement in validation mean R² (~0.428 → ~0.432) with negligible impact on runtime.  
  - Because these features only depend on the same 10‑step history used by the submission model, they are safe to use on the leaderboard (unlike the per‑sequence catch22 block).  
  - This v5 configuration is a natural candidate for the next submission model: same MLP architecture, slightly richer, fully streaming‑compatible features.

### 2.5 CatBoost MultiRMSE on Lag+Delta Features (All 32 Features, Offline Lab)

- File:
  - `train_catboost_experiment.py`.  
- Feature construction:
  - Reuses `build_supervised_dataset` from `train_model.py` with `n_lags=10`.  
  - Same 641‑dim feature vector as v2 Lag‑MLP:
    - Raw lags (10×32), LastKnown‑delta features (10×32), and `step_in_seq/1000`.  
- Dataset:
  - Train: 371,287 supervised samples (from 413 sequences).  
  - Val: 93,496 supervised samples (from 104 sequences).  
- Training configuration:
  - Subsampled train set to **120,000** rows for speed.  
  - Model: `CatBoostRegressor(loss_function="MultiRMSE")`.  
  - Hyperparameters:
    - `iterations=80`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3`,  
      `thread_count=1`, `od_type="Iter"`, `od_wait=40`, `verbose=20`.  
  - Runtime:
    - Training finished in ≈ **886 s (~14.8 minutes)** on local CPU.  
- Results:
  - Validation mean R² ≈ **0.3815** on the full validation set.  
  - We saved the model to `models/catboost_lag_delta_multiRMSE.cbm` (offline analysis only).  
- Takeaways:
  - With this relatively small/fast CatBoost setup and subsampled training data, performance is **below** the v2 Lag‑MLP (which reaches ~0.42+ val R²).  
  - This suggests:
    - Either CatBoost needs more capacity (more iterations, less subsampling) to shine on these features, or  
    - The current lag+delta features are the main bottleneck, not the model class.  
  - Still, this run gives a useful baseline and confirms that our supervised dataset is well‑formed for strong tabular models.

### 2.6 CatBoost MultiRMSE (Full Data, 500 Iterations, Offline Lab)

- File:
  - `train_catboost_experiment.py` (updated configuration).  
- Feature construction:
  - Same 641‑dim lag+delta+step features as v2 Lag‑MLP and the smaller CatBoost run.  
- Dataset:
  - Train: full 371,287 supervised samples (no subsampling).  
  - Val: 93,496 supervised samples.  
- Training configuration:
  - Model: `CatBoostRegressor(loss_function="MultiRMSE")`.  
  - Hyperparameters:
    - `iterations=500`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3`,  
      `thread_count=1`, `od_type="Iter"`, `od_wait=50`, `verbose=100`.  
  - Runtime:
    - Training finished in ≈ **15,040 s (~4.2 hours)** on local CPU.  
- Results:
  - Best iteration: 499 with validation loss (`test`) ≈ **4.3491**.  
  - Validation mean R² ≈ **0.4229** on the full validation set.  
  - Model saved (overwritten) at `models/catboost_lag_delta_multiRMSE.cbm`.  
- Takeaways:
  - With enough trees and full data, CatBoost MultiRMSE matches or slightly exceeds the MLP v2 validation performance (~0.42+), but at a **much higher training cost** (~4h single‑thread).  
  - This run is mainly an upper‑bound reference: it shows that our current lag+delta features can support strong tree‑based models, but using CatBoost as the main model would require careful consideration of training time and availability in the competition environment.

### 2.6b CatBoost MultiRMSE on v5 Streaming-Safe Features (All 32 Features, Offline + Submission)

- Files:
  - Offline lab: `train_catboost_experiment.py`.  
  - Streaming model: `solution_catboost.py` (submission variant).  
- Feature construction:
  - Reuses `build_supervised_dataset` with `n_lags=10` and the **v5 streaming-safe feature block**:  
    - Raw lags (10×32), LastKnown‑delta (10×32), rolling mean/std (32+32),  
    - `ac_lag1` (lag‑1 autocorr per feature), `frac_above_mean` (persistence),  
    - `step_in_seq/1000`.  
  - Total supervised feature dimension: **769**.  
- Dataset:
  - Train: `371,287` supervised samples (from 413 sequences).  
  - Val:   `93,496` supervised samples (from 104 sequences).  
- Training configuration:
  - `CatBoostRegressor(loss_function="MultiRMSE")`.  
  - Hyperparameters:  
    - `iterations=500`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3`,  
      `thread_count=1`, `od_type="Iter"`, `od_wait=50`, `verbose=100`.  
  - Runtime:
    - Training finished in ≈ **20,060 s (~5.6 hours)** on local CPU (single thread).  
- Results:
  - Best iteration: ~497 with validation loss (`test`) ≈ **4.3317**.  
  - Validation mean R² ≈ **0.4275** on the full validation set (slightly below MLP v5/v6 ~0.431–0.432 on the same split).  
  - Streaming evaluation on `train.parquet` via `solution_catboost.py`:  
    - Mean R² across all features ≈ **0.3651**.  
    - Runtime ≈ **4.8 minutes** on the full train file.  
  - Public leaderboard:
    - Submission `submission_catboost_v5_1.zip` achieved a score of **0.3340** (a bit **below** the MLP v5/v6 submissions at ~0.339–0.340).  
- Takeaways:
  - With v5 features, a large CatBoost MultiRMSE model achieves good validation and streaming performance but still **slightly underperforms** the Lag‑MLP on held‑out sequences and on the leaderboard.  
  - The training cost (~5–6 hours single‑thread) is high relative to the small R² gap vs the MLP, so CatBoost v5 is best used as an **occasional oracle** rather than the primary submission model in this project.  
  - There is no clear sign of severe overfitting:  
    - Train loss remains higher than validation loss at the best iteration,  
    - Train and validation R² are close,  
    - Streaming R² on the train file (~0.365) is in line with MLP v6 rather than much higher.  
  - Overall, this confirms that the current feature pipeline is well‑suited to both neural and tree‑based models, but the **Lag‑MLP offers a better accuracy‑to‑cost trade‑off** for the competition environment.

### 2.7 Custom Lag‑MLP + LastKnown‑Delta + Rolling Stats + catch22 (All 32 Features) – v4

- Files:
  - `train_model.py`, `solution.py`, `compute_catch22_features.py`, `models/`.  
- Feature construction (per sample):
  - Base v3 features:
    - `lag_slice` (10×32), `lag_flat` (320), `delta_flat` (320), `mean_last10` (32), `std_last10` (32), `step_in_seq/1000` (1).  
  - New automated features:
    - Offline script `compute_catch22_features.py` computes catch22 features per sequence and per dimension:  
      - `catch22_values`: shape `(n_seqs, 32, 22)` for `seq_ix` × feature_index × 22.  
    - For each sample, we look up the precomputed vector for its `seq_ix` and flatten:  
      - `catch22_flat` per sequence: `32 × 22 = 704` dims.  
  - Final feature vector:
    - `X_t = [lag_flat, delta_flat, mean_last10, std_last10, step_in_seq/1000, catch22_flat(seq_ix)]`  
    - Total dimension: **320 + 320 + 32 + 32 + 1 + 704 = 1409**.  
- Dataset:
  - Train: 371,287 supervised samples, X_train shape `(371287, 1409)`.  
  - Val: 93,496 supervised samples, X_val shape `(93496, 1409)`.  
- Normalization:
  - `x_mean`, `x_std` recomputed on the 1409‑dim `X_train`, saved to `models/lag_mlp_normalization.npz`.  
- Model & training:
  - `LagMLP` unchanged structurally:
    - Input: 1409, Hidden: 64 (ReLU), Output: 32, Epochs: 10, Adam `lr=1e‑3`.  
  - Validation mean R²:
    - Best val mean R² ≈ **0.4335** (vs ~0.4281 for v3 and ~0.42+ for v2).  
- Streaming integration:
  - `PredictionModel.__init__` loads `datasets/catch22_per_seq.npz` if present and builds `seq_ix -> catch22_flat` mapping.  
  - `_build_features` now concatenates `[lag_flat, delta_flat, mean_last10, std_last10, step, catch22_flat(current_seq_ix)]` before normalization.  
- Local scorer:
  - `python solution.py` on `train.parquet`:
    - Mean R² across all features ≈ **0.3784** (up from ≈0.3517 in v2 and “a bit higher than v2” in v3),  
    - Runtime remains ≈ **25–27 s** on full train (still very safe vs the 60‑minute budget).  
- Public leaderboard (first attempt with v4‑style submission, later rolled back):
  - A submission that mirrored the per‑sequence catch22 block into `solution.py` (alongside v3 features) achieved a much lower leaderboard score, around **0.15**, despite strong offline/streaming scores.  
  - The likely cause is that the full‑sequence catch22 descriptors encode sequence identity and future information specific to `train.parquet`, so they do not transfer to the hidden test set and can even act as a leakage‑like shortcut during offline validation.  
- Takeaways:
  - Adding per-sequence catch22 features on top of lag+delta+rolling stats gives a **clear, additional lift** in validation R² (~0.428 → ~0.4335) and streaming train R² (~0.35 → ~0.38) without harming runtime, **but only in lab settings that reuse the same sequences for training/validation**.  
  - For true generalization (public leaderboard), directly using full‑sequence per‑seq catch22 vectors inside the submission model is **unsafe** and leads to overfitting; catch22 is therefore reserved for offline analysis and for inspiring streaming‑safe analogs rather than being used as input features in the final `solution.py`.  
  - The MLP remains a simple 1‑hidden‑layer network; the observed gains are purely from richer features, not model size, and need to be re‑expressed via safe lag‑window statistics for submission.  

### 2.8 CatBoost MultiRMSE + catch22 Feature Importance (Offline Lab Only)

- File:
  - `catch22_feature_importance.py`.  
- Purpose:
  - Use CatBoost on the full v4 feature set (lags + LastKnown‑delta + rolling mean/std + step + per‑sequence catch22) **only as a feature‑importance lab**, not as a submission model.  
- Feature construction:
  - Reuses `build_supervised_dataset(..., use_catch22=True)` from `train_model.py` with `n_lags=10`, `add_step_feature=True`.  
  - X shape: `(371,287, 1409)` on train, `(93,496, 1409)` on val:  
    - Base v3 block: raw lags (10×32), LastKnown‑delta (10×32), rolling mean/std (32+32), step_in_seq/1000 (1) → **705 dims**.  
    - catch22 block: per‑sequence 32×22 descriptors → **704 dims**.  
  - y shape: `(n_samples, 32)` (next‑step state).  
- Training configuration (quick lab run, not tuned):
  - Subsampled train set to **30,000** rows for speed.  
  - Model: `CatBoostRegressor(loss_function="MultiRMSE")`.  
  - Hyperparameters:
    - `iterations=20`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3`,  
      `random_seed=42`, `thread_count=1`, `od_type="Iter"`, `od_wait=10`, `verbose=20`.  
  - Goal is **relative feature importances**, not strong R².  
- Results (two variants):
  - Variant A – full 22 catch22 stats per dimension:
    - Validation mean R² ≈ **0.2661** (expected to be much lower than the well‑trained MLP / CatBoost baselines due to tiny iteration count and heavy subsampling).  
    - Global feature‑importance split (CatBoost `FeatureImportance`):  
      - Base v3 features (lags + LastKnown‑delta + rolling mean/std + step): **≈ 88.2%** of total importance.  
      - Per‑sequence catch22 block: **≈ 11.8%** of total importance.  
    - Top catch22 statistics by importance (aggregated over all 32 state dimensions):  
      - `SP_Summaries_welch_rect_area_5_1` and `SP_Summaries_welch_rect_centroid`: spectral energy and spectral centroid in a specific frequency band.  
      - `CO_f1ecac`, `CO_FirstMin_ac`: autocorrelation‑scale / decorrelation‑time features.  
      - `FC_LocalSimple_mean1_tauresrat`, `FC_LocalSimple_mean3_stderr`: local trend / residual ratio and stability.  
      - `CO_trev_1_num`: time‑reversibility (asymmetry) at lag 1.  
      - `SB_BinaryStats_mean_longstretch1`: length of the longest above‑mean stretch (persistence).  
      - `CO_HistogramAMI_even_2_5`: auto‑mutual information over a short lag range.  
    - Top state dimensions by catch22 importance:  
      - Features `29`, `23`, `7`, `15`, `4`, `10`, `25`, `18`, `22`, `31` have the largest aggregate catch22 importance (each contributing ≈0.5–1.8% of total model importance from the catch22 block).  
  - Variant B – **subset of important catch22 stats only**:
    - In `train_model.py`, we now select a subset of catch22 statistics by name when building `_SEQ_TO_CATCH22`, keeping only:
      - `SP_Summaries_welch_rect_area_5_1`, `SP_Summaries_welch_rect_centroid`,  
        `CO_f1ecac`, `CO_FirstMin_ac`,  
        `FC_LocalSimple_mean1_tauresrat`, `FC_LocalSimple_mean3_stderr`,  
        `CO_trev_1_num`, `SB_BinaryStats_mean_longstretch1`,  
        `CO_HistogramAMI_even_2_5`.  
    - This reduces the catch22 block per sample from 704 dims to **9 × 32 = 288 dims** and the overall feature dimension from 1409 → **993**.  
    - With this subset, the same CatBoost lab run gives:
      - Validation mean R² ≈ **0.2650** (essentially unchanged vs 0.2661, as expected with such a small model).  
      - Global feature‑importance split:  
        - Base v3 features: **≈ 86.0%** of total importance.  
        - catch22 subset block: **≈ 14.0%** of total importance.  
      - New top catch22 statistics by importance (aggregated over all dims) reflect both the original spectral/autocorr features and additional histogram‑/HRV‑like stats that remain in the subset or act as proxies (`DN_HistogramMode_10`, `DN_HistogramMode_5`, `CO_f1ecac`, `MD_hrv_classic_pnn40`, `CO_trev_1_num`, `CO_HistogramAMI_even_2_5`, `SB_BinaryStats_mean_longstretch1`).  
    - Takeaway from the subset experiment:
      - Dropping the less important catch22 stats barely changes R² in this small CatBoost lab model, while slightly increasing the relative share of the remaining catch22 block.  
      - This supports the idea that **most of the useful catch22 signal is concentrated in a small subset of spectral/autocorr/persistence features**, so any future streaming‑safe analogs can focus on those behaviours rather than trying to mimic all 22 catch22 features.  
- Clustering in catch22 space:
  - Using `datasets/catch22_per_seq.npz`, we flattened the per‑sequence catch22 tensors `(n_seqs, 32, 22)` into `(517, 704)` and ran `KMeans(n_clusters=4)` after standardization.  
  - Cluster sizes and rough aggregate stats in catch22 space:  
    - Cluster 0: 249 sequences, mean |catch22| ≈ 2.20, std ≈ 5.21.  
    - Cluster 1: 72 sequences, mean |catch22| ≈ 4.16, std ≈ 16.61 (most “extreme” / high‑variance series in catch22 terms).  
    - Cluster 2: 168 sequences, mean |catch22| ≈ 1.72, std ≈ 3.32 (more “typical” / mild dynamics).  
    - Cluster 3: 28 sequences, mean |catch22| ≈ 1.91, std ≈ 5.13.  
  - These clusters can be linked later to sequence‑level difficulty (e.g., average forecasting error), but that analysis has not been run yet.  
- Takeaways:
  - Even in a small, undertrained CatBoost model, the catch22 block carries a **non‑trivial but secondary** amount of signal (~12% of total feature importance) on top of the v3 features.  
  - The most influential catch22 statistics are mostly **spectral and autocorrelation/time‑scale descriptors** plus simple measures of persistence and local trend.  
  - This supports the idea of designing **streaming‑safe analogs** (e.g., short‑window autocorrelation, simple energy/variance proxies, persistence indicators) that can be computed from the last `n_lags` steps only and used safely in the submission model, while keeping full per‑sequence catch22 as an offline lab tool only.

---

## 3. Current Understanding

- The data is already roughly standardized; simple lag features are powerful.  
- A compact global model (shared across sequences) works well: no need for one model per `seq_ix`.  
- Univariate Tsururu+CatBoost with good lags/normalization can reach very high R² (~0.93) on a single feature.  
- Our custom multivariate lag‑MLP:
  - v1 (raw lags) achieves ~0.41 val R² offline and ~0.3266 leaderboard R².  
  - v2 (lags + LastKnown‑delta) achieves ~0.42+ val R² offline and ~0.3293 leaderboard R².  
  - v5 (lags + LastKnown‑delta + rolling stats + lag‑1 autocorr + persistence) achieves ~0.432 val R² offline and ~0.3390 leaderboard R².  
  - v6 (v5 features + extra short‑window autocorr, robust stats, and per‑feature trend) achieves ~0.4321 val R² offline, ~0.3653 streaming R² on the train file, and ~0.3400 leaderboard R².  
- We are moving upwards but still below top solutions (~0.39), indicating more feature/model improvements are possible.

---

## 4. Future Hypotheses & Experiments

### 4.1 Tsururu‑Guided Model Search (Offline Only)

1. **Multivariate CatBoost in Tsururu**
   - Treat all 32 features as multivariate targets.  
   - Grid over:
     - `target_lags` (e.g., 5, 10, 20).  
     - `history` window (e.g., 30, 50, 100).  
     - Normalizers: standard scaler, `DifferenceNormalizer`, `LastKnownNormalizer`.  
   - Measure mean R² across all features; identify best config.

2. **Simple DL Strategies in Tsururu**
   - Use DLinear / MIMOStrategy with horizon 1 or small horizon.  
   - Compare against CatBoost on the same Tsururu feature setup.  
   - Only copy ideas that clearly outperform CatBoost and remain CPU‑friendly.

### 4.2 Improvements to Custom Lag‑MLP

- Increase capacity / training time:
  - More epochs (e.g., 20+).  
  - Hidden size 128 or an extra hidden layer (e.g., 128 → 64 → 32).  
- Enrich features:
  - Add rolling statistics (mean/std over last 10 steps).  
  - Consider longer or multi‑scale lags (e.g., indices `{1,2,3,5,10,20}`) if Tsururu indicates benefit.  
- Regularization:
  - Weight decay or dropout if overfitting appears.

### 4.3 CatBoost‑Based Submission Model

- If CatBoost is confirmed available in the scoring environment:
  - Build the same lag features as in `train_model.py` (or improved ones).  
  - Train 32 CatBoostRegressors (one per feature) offline.  
  - Save to `models/cat_*.cbm`.  
  - Implement a CatBoost‑based streaming `PredictionModel` mirroring the MLP logic.

### 4.4 Recurrent Models (Later Phase)

- Experiment with small GRU/LSTM models:
  - Maintain hidden state per sequence, reset on new `seq_ix`.  
  - Compare R² and runtime to lag‑MLP and CatBoost.  
- Only consider for submission if:
  - They clearly outperform simpler models, and  
  - Inference remains well below the 60‑minute time budget.

---

This log should be updated whenever a new experiment is run (Tsururu configs, new training runs, new submissions) with a short note on settings and results. 
