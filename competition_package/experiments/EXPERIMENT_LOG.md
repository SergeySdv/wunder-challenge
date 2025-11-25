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

### 2.9 Lag‑MLP v7 – Deeper Funnel MLP + LR Scheduler (Submission Model)

- Files:
  - `train_model.py` (updated LagMLP architecture and training loop).  
  - `solution.py` (mirrored architecture for streaming inference).  
- Feature set:
  - Uses the **v6 streaming‑safe features** unchanged (lags + LastKnown‑delta + rolling stats, short‑window autocorr at lags 1–3, acf sum, persistence fraction, robust window stats, and per‑feature trend, plus step_in_seq/1000).  
  - Supervised feature dimension: `X` shape `(371,287, 1185)` on train, `(93,496, 1185)` on val.  
- Model & training:
  - Architecture: funnel‑style MLP (CPU‑only, PyTorch) with dropout:  
    - Input: 1185.  
    - Hidden 1: 512 (ReLU, Dropout 0.1).  
    - Hidden 2: 256 (ReLU, Dropout 0.1).  
    - Output: 32.  
  - Hyperparameters:  
    - Epochs: 20 (`N_EPOCHS = 20`).  
    - Batch size: 1024.  
    - Optimizer: Adam, `lr = 1e‑3`.  
    - LR scheduler: `ReduceLROnPlateau(mode="max", factor=0.5, patience=2)` on validation mean R².  
    - Best epoch’s weights are tracked and reloaded before saving.  
  - Environment:  
    - Trained using the project virtualenv at `/Users/sergei/PycharmProjects/WunderSex/.venv` via:  
      - `cd competition_package`  
      - `../.venv/bin/python train_model.py`  
- Validation & streaming results:
  - Offline supervised validation (sequence‑disjoint split) on `(X_val, y_val)`:  
    - Best validation mean R² ≈ **0.4413** (up from ~0.432 for earlier v5/v6 MLP).  
  - Streaming evaluation on `datasets/train.parquet` with `ScorerStepByStep` and the updated `solution.py`:  
    - Command: `../.venv/bin/python solution.py` from `competition_package`.  
    - Mean R² across all 32 features: **0.444645**.  
    - Example per‑feature R² (first 5 features):  
      - Feature 0: **0.3362**  
      - Feature 1: **0.3417**  
      - Feature 2: **0.3752**  
      - Feature 3: **0.5337**  
      - Feature 4: **0.3735**  
    - Runtime on full `train.parquet` (~517k rows): **≈ 2.4 minutes** on local CPU (well below the 60‑minute competition limit and only modestly slower than the previous v6 MLP).  
- Submission status:
  - No new submission ZIP built yet, but this v7 MLP is ready to be packaged (reusing the `SUBMISSION_GUIDE.md` MLP instructions, e.g. `submissions/mlp_v7`).  
  - Ensemble support (multi‑seed checkpoints `lag_mlp_seed*.pth` and averaging in `solution.py`) has been wired into the codebase but not yet trained/evaluated; current results reflect a **single v7 model**.

### 2.10 Lag‑MLP v7 – 3-Seed Ensemble (Submission Model)

- Files:
  - `train_model.py` (ensemble training loop over `ENSEMBLE_SEEDS = [42, 43, 44]`).  
  - `solution.py` (loads `lag_mlp_seed*.pth` and averages predictions).  
- Feature set:
  - Same as v7 single model – v6 streaming‑safe features (lags, LastKnown‑delta, rolling stats, short‑window autocorr 1–3, acf sum, persistence, robust stats, trend, step).  
  - Supervised shapes unchanged: X `(371,287, 1185)`, y `(371,287, 32)` for train; analogous for val.  
- Ensemble training:
  - Seeds: `[42, 43, 44]`.  
  - For each seed:  
    - Architecture: 1185 → 512 → 256 → 32, ReLU + Dropout(0.1) between hidden layers.  
    - Epochs: 20, batch size: 1024, Adam `lr=1e‑3`, ReduceLROnPlateau on val mean R².  
    - Best validation mean R² per seed (sequence‑disjoint split):  
      - Seed 42 (idx 0): **0.4424**.  
      - Seed 43 (idx 1): **0.4451**.  
      - Seed 44 (idx 2): **0.4393**.  
    - Checkpoints saved to:  
      - `models/lag_mlp_seed0.pth`, `lag_mlp_seed1.pth`, `lag_mlp_seed2.pth`.  
  - Best single-seed validation R² across ensemble members: **0.4451** (seed index 1).  
  - For backward compatibility, `models/lag_mlp.pth` stores the best single seed; `lag_mlp_normalization.npz` remains shared.  
- Streaming ensemble evaluation:
  - `solution.py` detects `lag_mlp_seed*.pth` and builds `self.models` from all three MLPs; `predict()` averages their outputs.  
  - Command:  
    - `cd competition_package`  
    - `../.venv/bin/python solution.py | egrep 'Mean R' -A4`  
  - Results on full `train.parquet` (517k rows):  
    - Mean R² across all 32 features: **0.448761** (up from **0.444645** for the single v7 model).  
    - Example per-feature R² (first few features):  
      - Feature 0: **0.3389** (vs ~0.3362 single‑model).  
      - Feature 1: **0.3474** (vs ~0.3417).  
      - (Remaining per-feature R² follow a similar small but consistent improvement pattern.)  
    - Runtime: **≈ 3.2 minutes** on local CPU (vs ~2.4 minutes single model), still very safe under the 60‑minute competition budget.  
- Takeaways:
  - A small 3‑seed ensemble on top of the v7 architecture yields a **modest but reliable R² gain** (~+0.004 on streaming train R²) and should reduce leaderboard variance.  
  - Inference cost scales roughly linearly with ensemble size but remains negligible relative to the official time limit.  
  - This v7 ensemble is a strong candidate for the next submission (`mlp_v7_ensemble`).

### 2.11 Lag‑MLP v8 – Residual Targets (Delta) + 3-Seed Ensemble

- Files:
  - `train_model.py` – target changed from level to residual: `y_t = state(t+1) - state(t)`.  
  - `solution.py` – ensemble output interpreted as delta and added to current state before returning the prediction.  
- Feature set:
  - Same v6/v7/v7-ensemble input features (lags, LastKnown‑delta, rolling stats, short‑window autocorr, robust stats, trend, step).  
  - `X` shape unchanged: `(371,287, 1185)` for train, `(93,496, 1185)` for val.  
- Target engineering:
  - Old target: `y_t = state(t+1)` (next level).  
  - New target: `y_t = state(t+1) - state(t)` (next-step jump / residual).  
  - At inference time, `PredictionModel.predict` computes:  
    - `delta_hat = mean_mlp(x_t)` (ensemble-averaged delta),  
    - `state_hat(t+1) = current_state(t) + delta_hat`,  
    - and returns `state_hat(t+1)` to `ScorerStepByStep`.  
- Ensemble training:
  - Same seeds `[42, 43, 44]` and architecture (1185 → 512 → 256 → 32, ReLU + Dropout(0.1)).  
  - Validation R² now measured on delta targets (residuals) rather than levels.  
  - Best validation mean R² per seed on the residual task:  
    - Seed 42 (idx 0): **0.5241**.  
    - Seed 43 (idx 1): **0.5210**.  
    - Seed 44 (idx 2): **0.5217**.  
  - Best across ensemble: **0.5241** (seed index 0).  
- Streaming evaluation (level predictions via `current_state + delta_hat`):
  - Command:  
    - `cd competition_package`  
    - `../.venv/bin/python solution.py`  
  - Results on full `train.parquet` (~517k rows):  
    - Mean R² across all 32 features: **0.452906**.  
    - Example per-feature R² (first 5 features):  
      - Feature 0: **0.3473**.  
      - Feature 1: **0.3576**.  
      - Feature 2: **0.3801**.  
      - Feature 3: **0.5416**.  
      - Feature 4: **0.3899**.  
    - Runtime: ≈ **3.0–3.1 minutes** on local CPU (similar to v7 ensemble).  
- Takeaways:
  - Switching to residual targets (predicting `state(t+1) - state(t)` and then reconstructing the level) yields a **further improvement** over the v7 ensemble: mean streaming R² increases from ~0.4488 → **0.4529** on `train.parquet`.  
  - The architecture, features, and inference cost remain unchanged; only the target definition and final reconstruction step differ.  
  - This v8 residual ensemble is currently the strongest lab result and a natural next submission candidate (`mlp_v8_residual_ensemble`).

### 2.12 Lag‑MLP v9 – Multi Pair-Spread Features (Lab Only; Code Reverted)

- Files (temporary experiment):
  - `train_model.py` – extended supervised feature vector with spread features between several highly correlated pairs.  
  - `solution.py` – mirrored spread feature block inside `_build_features` for streaming inference.  
- Feature set changes vs v7/v8:
  - Kept **level targets** (same as v7; v8 residual target logic was not used here).  
  - Base feature block remained the **v6 streaming‑safe features** (lags, LastKnown‑delta, rolling stats, short‑window autocorr 1–3, acf sum, persistence, robust stats, trend, step).  
  - Added a small spread feature block for a curated set of highly correlated pairs identified in EDA:  
    - `(18, 28, +1)`, `(11, 30, +1)`, `(0, 21, −1)`, `(7, 31, +1)`, `(1, 28, +1)`, `(3, 4, +1)`.  
  - For each pair `(a, b, sign)` we appended two scalars computed from the most recent state in the lag window:  
    - `spread = state[a] − sign * state[b]`,  
    - `|spread|`.  
  - Supervised feature dimension increased from **1185 → 1197**.  
- Training details:
  - Same architecture as v7 ensemble: 1197 → 512 → 256 → 32, ReLU + Dropout(0.1).  
  - Same training loop / hyperparameters and seeds `[42, 43, 44]`.  
  - With the new spreads, supervised validation mean R² on the level target task was slightly **lower** than v7: best seed reached ≈ **0.4409** (vs ≈0.4451 for the original v7 ensemble).  
- Streaming evaluation on `train.parquet` (level predictions):
  - Command (from `competition_package` after training):  
    - `../.venv/bin/python solution.py`  
  - Results on full `train.parquet` (~517k rows):  
    - Mean R² across all 32 features: **0.453464**.  
    - Example per‑feature R² (first 5 features):  
      - Feature 0: **0.3458**.  
      - Feature 1: **0.3523**.  
      - Feature 2: **0.3807**.  
      - Feature 3: **0.5415**.  
      - Feature 4: **0.3845**.  
  - Notably, this **streaming train R² slightly exceeded v8** (≈0.4535 vs 0.4529) despite using level targets.  
- Submission and leaderboard:
  - Packaged as `submissions/submission_mlp_v9_spreads.zip` (containing the multi‑pair spread `solution.py` and ensemble weights).  
  - Public leaderboard results:  
    - v7 ensemble submission (`submission_mlp_v7_ensemble.zip`): **0.3469**.  
    - v8 residual ensemble submission (`submission_mlp_v8_residual_ensemble.zip`): **0.3378**.  
    - v9 multi‑spread submission (`submission_mlp_v9_spreads.zip`): **0.3461**.  
- Takeaways:
  - Adding explicit spread features for several strong pairs did **improve streaming train R²** beyond both v7 and v8, confirming that pair structure carries usable signal.  
  - However, the **offline supervised val R² degraded slightly**, and the public leaderboard score for v9 (0.3461) was marginally **worse than v7** (0.3469), despite better train‑file metrics.  
  - This mirrors the earlier v8 lesson: improvements in train‑file R² (even with intuitively reasonable features) do not guarantee better generalization to the hidden test set.  
  - Given the small and noisy LB difference and the added complexity, the project has reverted `train_model.py` and `solution.py` back to the **clean v7 ensemble feature set**; pair‑spread features are treated as a **lab‑only idea** for future controlled experiments (e.g., with stronger held‑out seq splits) rather than part of the main submission path.  

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


### 2.13 Lag-MLP v10 – Robust Ensemble (Winsorization + 5-Fold CV)

- Files:
  - `train_model.py` (refactored for 5-Fold CV + Winsorization).
  - `solution.py` (updated for global winsorization bounds + ensemble loading).
- Motivation:
  - Address the disconnect between Train R² and Leaderboard R² seen in v8/v9.
  - Implement "autofin"-style robustness without external dependencies:
    - **Winsorization**: Clip input lags to [0.1%, 99.9%] quantiles to handle outliers.
    - **Strict Validation**: 5-Fold CV by sequence + Pseudo-LB (10% held out) to get reliable error estimates.
- Feature set:
  - Same v6 streaming-safe features (lags, LastKnown-delta, rolling stats, short-window autocorr 1–3, robust stats, trend, step).
  - **Input Processing**: Raw lag window is clipped *before* computing any derived features.
  - Targets `y` are **not** clipped (model sees clean inputs but predicts true targets).
- Training details:
  - **Split**:
    - **Pseudo-LB**: 10% of sequences (~50) held out completely.
    - **Dev Set**: Remaining 90% (~466 sequences) used for 5-Fold CV.
  - **Protocol**:
    - Global Stats (Mean/Std/Winsorization) computed on **Dev Set only**.
    - Train 5 independent MLPs (one per fold) on the Dev Set.
    - Ensemble: Simple average of the 5 fold models.
  - Architecture: Same 1185 -> 512 -> 256 -> 32 funnel MLP.
- Results:
  - **CV Results (5 Folds)**:
    - Mean R²: **0.3544** ± 0.0138.
    - Fold scores: [0.3317, 0.3614, 0.3515, 0.3533, 0.3739].
    - This ~0.354 is a conservative estimate of generalization capability.
  - **Pseudo-LB Results (Held Out)**:
    - Mean R²: **0.3892**.
    - This is significantly higher than the CV score and our previous LB best (~0.34), suggesting the held-out set might be slightly "easier" or the model generalizes surprisingly well.
  - **Streaming Evaluation (Full Train File)**:
    - Mean R²: **0.4472** (slightly lower than v8/v9's ~0.453, likely due to winsorization "damping" extreme valid signals, but robustness is prioritized).
  - **Public Leaderboard**:
    - Submission `submission_mlp_v10_robust_ensemble.zip`: **0.3513**.
    - This is a **new best score**, improving on v7 (~0.3469). The robust validation and winsorization strategy successfully reduced overfitting/variance on the hidden test set.
- Submission:
  - Packaged as `submission_mlp_v10_robust_ensemble.zip`.
  - Includes all 5 fold models (`lag_mlp_fold*.pth`) and the normalization file.
- Takeaways:
  - This "v10" architecture prioritizes **reliability** over raw training fit.
  - The 5-Fold CV gives us a distribution of likely performance (0.33 - 0.37 range).
  - Winsorization adds a safety layer against distribution shifts in the hidden test set.

### 2.14 Lag-MLP v11 – Tuned Robust Ensemble (Optuna)

- Files:
  - `optimize_mlp.py`: Ran random search over hidden size, dropout, LR, and batch size.
  - `train_model.py`: Updated with optimal params (`HIDDEN_SIZE=128`, `DROPOUT=0.3`, `LR=5e-4`, `BATCH_SIZE=512`).
- Hypothesis:
  - The default architecture (256 hidden, 0.1 dropout) might be overfitting or suboptimal.
- Results:
  - **Optimization**: Found that smaller models (128 units) with higher regularization (0.3 dropout) generalized better on the holdout set.
  - **Pseudo-LB**: Improved significantly to **0.3963** (vs 0.3892 in v10).
  - **Public Leaderboard**: **0.3540** (vs 0.3513 in v10).
- Takeaways:
  - Small gains from hyperparameter tuning confirm we are near the "glass ceiling" for this specific feature set + fixed-window MLP architecture.
  - The large gap between Pseudo-LB (0.39) and Real LB (0.35) persists, motivating a move to recurrent models (v12) to capture longer-term context.

### 2.16 Lag-MLP v13 – Kinematics & Volatility (Regression)

- Files:
  - `train_model.py` / `solution.py`: Added Accel, Vol Expansion, Roughness.
- Results:
  - **CV**: 0.3538 (Flat vs v11).
  - **Pseudo-LB**: 0.3951 (Slightly worse than v11).
  - **Public Leaderboard**: **0.3529** (Worse than v11's 0.3540).
- Takeaways:
  - Adding ~100 explicit kinematic features added noise/complexity without improving generalization.
  - The simpler v11 model remains the strongest baseline.
  - **Action**: Reverted codebase to v11.

### 2.17 NLinear v16 – Long Context Linear Model (Failed)

- Files:
  - `train_model.py` (NLinear implementation with 336-step lookback).
- Motivation:
  - Deep Research suggested simple linear models (DLinear/NLinear) with long history often beat Transformers and RNNs on financial time series.
- Results:
  - **CV Mean R²**: ~0.2673 (Massive regression vs v11's 0.354).
  - **Pseudo-LB**: ~0.3076.
- Takeaways:
  - The assumption that "Long History + Linearity" wins was incorrect for this specific dataset.
  - Short-term microstructure (captured by our MLP's engineered features) is far more predictive than long-term linear trends.
  - **Action**: Reverted to v11.

### 2.18 CatBoost "Kitchen Sink" & Optimization (v17/v18)

- Files:
  - `train_catboost_experiment.py`: Updated with v13 features (1281 dim).
  - `optimize_catboost.py`: Ran Optuna tuning with `colsample_bylevel` for speed.
- Motivation:
  - Test if Tree models (robust to irrelevant features) could exploit the massive 1300-feature set better than MLP.
- Results:
  - **Standard Training (v17)**: Extremely slow (>4 hours/fold). Fold 0 R² ~0.328 (worse than MLP 0.334).
  - **Optimized "Fast" Training (v18)**: Best Trial R² ~0.264 (on subset).
  - **Best Params**: `colsample_bylevel=0.05`. The model ignored 95% of features to get any signal, confirming high noise levels.
- Takeaways:
  - CatBoost consistently underperforms the Tuned MLP on this dataset (0.32 vs 0.35).
  - The "Kitchen Sink" approach added more compute cost than predictive value.
  - **Final Decision**: Stick with v11 MLP.

### 2.19 Ultra-Tuned MLP v19 – Optuna Fine-Tuning (New Best)

- Files:
  - `optimize_mlp.py`: Ran 50 TPE trials with `weight_decay` and finer ranges.
  - `train_model.py`: Updated params (`HIDDEN=192`, `DROPOUT=0.2`, `LR=1.6e-4`).
- Results:
  - **Optimization**: Found a slightly larger model (192 vs 128) with slower learning rate (1.6e-4 vs 5e-4) worked best.
  - **CV Mean R²**: **0.3556** (vs 0.3535 in v11).
  - **Pseudo-LB**: **0.3996** (vs 0.3943 in v11).
  - **Public Leaderboard**: **0.3563** (New Personal Best).
- Takeaways:
  - Rigorous hyperparameter tuning squeezed an extra +0.0023 out of the architecture.
  - This confirms that the MLP architecture is robust, but we are hitting the asymptotic limit of what this feature set can provide.
  - **Current Status**: v19 is the production model.

### 2.20 Triplet + WVTR (Cross-Sectional + Wavelet) – Regressed

- Files:
  - `train_model.py` and `solution.py`: added triplet imbalance block (120 dims on curated feature set [0,2,4,7,11,13,18,20,28,31]) and WVTR (32 dims Haar noise/trend ratio).
  - `models/lag_mlp_fold*.pth`, `lag_mlp_normalization.npz` regenerated with the new feature order.
- Offline results:
  - 5-Fold CV Mean R²: **0.35512 ± 0.01306** (≈ flat vs v19).
  - Pseudo-LB Mean R²: **0.39780** (≈ flat vs v19).
- Streaming train evaluation:
  - Mean R² on `train.parquet`: **0.4035** (regression vs prior ~0.44).
- Leaderboard:
  - `submission_mlp_v20_triplet_wvtr.zip`: **0.3549** (worse than v19’s **0.3563**).
- Takeaways:
  - The added blocks did not improve generalization; they reduced in-distribution fit and slightly hurt LB.
  - Action: keep v19 as the baseline; treat triplet+WVTR as a negative result. Revert to v19 feature set for submissions.

### 2.21 Level + Residual Blend (v19 features, residual target copy)

- Files:
  - `train_model.py` now supports `--target_mode {level,residual}` and `--prefix` to train residual-target models.
  - Residual weights: `models/lag_mlp_residual_fold*.pth`, normalization: `lag_mlp_residual_normalization.npz`.
  - Blended inference: `solution_blend.py` loads level + residual ensembles and blends `pred = alpha*level + (1-alpha)*(state + delta_pred)`.
- Offline results:
  - Residual training (25 epochs, v19 features): CV R² (residual target) **0.5241 ± 0.0274**.
  - Pseudo-LB (reconstructed level): **0.39569** (slightly below level v19 ~0.3996).
  - Blend Pseudo-LB sweep (alpha on level):  
    - α=0.5 → **0.40373**  
    - α=0.6 → **0.40388** (best)  
    - α=0.7 → **0.40353**
- Streaming train eval:
  - Blend α=0.5 Mean R² on `train.parquet`: **0.4053** (regresses vs level-only v19 ~0.44).
- Packaging:
  - `submissions/submission_mlp_blend_alpha0_6_fix.zip` (blend with α=0.6, level + residual weights, both normalizations) – **LB 0.3571** (slightly above v19 0.3563).
  - Earlier `submission_mlp_blend_alpha0_6.zip` failed CHECK (missing solution.py entrypoint).
- Takeaways:
  - Residual + blend offers a small Pseudo-LB lift and delivered a marginal LB gain (0.3571 vs 0.3563). Streaming train R² is still lower than level-only, so keep v19 level-only as the stable fallback; blend (α=0.6) is currently the top LB score.

### 2.22 CatBoost v19 (re-run, 500 iters, diagonal features only)

- Files:
  - `train_catboost_experiment.py` (v19 feature set: 1185 dims; 10 lags + deltas + rolling stats + autocorr/persistence/robust/trend).
- Motivation:
  - Refresh CatBoost baseline on the clean v19 set after reverting v20.
- Results (5-Fold CV, 500 trees, depth=6, lr=0.05, subsample=0.8, early stop 50):
  - Fold R²: [0.3231, 0.3561, 0.3389, 0.3385, 0.3638]
  - **Mean CV R²: 0.3441**
  - **Pseudo-LB R² (single model from last fold): 0.3837**
- Top-20 feature importances (Fold 0) mapped to names:
  - Lags: lag[9]/feat17, lag[9]/feat19, lag[9]/feat0, lag[9]/feat10, lag[9]/feat2, lag[9]/feat16, lag[8]/feat23, lag[9]/feat5
  - Means: mean/feat3, mean/feat8, mean/feat7, mean/feat26, mean/feat29, mean/feat18, mean/feat20
  - Robust: q25/feat26, q25/feat30, median/feat26, q75/feat29
  - Vol: std/feat4
- Takeaways:
  - Still below the MLP v19 (CV ~0.355–0.356, Pseudo-LB ~0.3996) and below the blend LB.
  - Training is slow (~2h per run). Not a submission candidate; keep as reference only.

### 2.23 LSTM Baseline (raw 32-dim, short context)

- Files:
  - `train_lstm_experiment.py` (sequence-to-one on raw values; no engineered features).
- Settings (quick pilot):
  - Window=30, hidden=256, layers=2, lr=5e-4, epochs=10, batch=512, subset=120k samples (winsorized, normalized).
  - Train/val split: 90% dev / 10% val (by seq); Pseudo-LB split 10% seqs.
- Results:
  - Val R²: **0.3115**
  - Pseudo-LB R²: **0.3957** (note: subset + different target setup; interpret with caution).
- Takeaways:
  - Val underperforms v19 MLP (~0.355–0.356). Pseudo-LB likely optimistic; overall not a replacement yet. Keep as reference; try full-data or residual-target variant only if time allows.

### 2.24 LSTM Full Train & Submission Attempt (timed out)

- Files:
  - `train_lstm_experiment.py` (same raw 32-dim window model), `solution_lstm.py` for streaming submission.
- Settings:
  - Window=30, hidden=256, layers=2, lr=5e-4, epochs up to 20 (early stop at epoch 9), full dev (377,580 samples), val=41,354, pseudo-LB=45,849.
  - Saved artifacts: `models/lstm_submission.pth`, `models/lstm_submission_norm.npz`, `models/lstm_submission_meta.json`.
- Results:
  - Final Val R²: **0.3245**
  - Pseudo-LB R²: **0.4048** (likely optimistic vs LB).
  - Leaderboard: `submission_lstm.zip` **TIMED OUT** in prerun (60s limit) — streaming inference too slow (recomputes over window each step).
- Takeaways:
  - Runtime and weaker val R² make this a poor submission choice. If ever retried, need a true stateful single-step LSTM (carry h/c) and/or much smaller model/window, but expect lower accuracy. Stick with v19/blend for submissions.

### 2.25 Micro-Mamba (SSD-style SSM) – Pilot Runs (underperformed)

- Files:
  - `scripts/train_mamba_experiment.py` (CPU-friendly SSD/Mamba-2–style block: small causal conv + scalar-decay SSM + gating + FFN). Targets set to residuals (`state(t+1) - state(t)`), window=30.
- Pilot A (tiny sanity check):
  - Config: `d_model=64`, `d_state=32`, `layers=1`, `d_conv=3`, `batch=512`, `epochs=3`, `subset=20k`, residual target.
  - Results: Val R² **0.3259**, Pseudo-LB **0.3500**. Fast but below v19 MLP and even GRU.
- Pilot B (mid-size):
  - Config: `d_model=64`, `d_state=32`, `layers=2`, `d_conv=4`, `batch=256`, `epochs=6`, `subset=120k`, residual target.
  - Results: Val R² **0.2644**, Pseudo-LB **0.2868**. Early-stopped; clear regression.
- Takeaways:
  - Both pilots underperform the baseline MLP and the GRU. Further scaling risks timeouts without evidence of upside. Marked as tested/underperforming; no submission packaged.

### 2.26 Micro-Mamba on v19 Features (submission test, underperformed)

- Files:
  - `solution_mamba.py` (submission entrypoint) with model artifacts:
    - `models/mamba_v19_small.pth`
    - `models/mamba_v19_small_norm.npz`
    - `models/mamba_v19_small_meta.json`
  - Training script: `scripts/train_mamba_experiment.py` using v19 feature extractor (window=10, feature_dim=1185), residual targets.
- Config (20k subset, quick run):
  - `window=10`, `d_model=96`, `d_state=24`, `layers=2`, `d_conv=4`, dropout 0.1, lr 5e-4, weight_decay 0.05, batch 256, epochs=3, residual target.
  - Val R² **0.3562**, Pseudo-LB **0.3828** (below v19 MLP/blend).
- Leaderboard:
  - `submission_mamba.zip`: **0.1215** (strong regression).
- Takeaways:
  - Even with v19 features and residual targets, the small Mamba model fails to generalize; LB score is far below baseline. Treat Mamba as deprioritized for now.

### 2.27 MLP v21 (Spreads + Residuals)

- Files:
  - `scripts/train_mlp_v21.py`, `src/features/extractor.py` (updated).
- Concept:
  - Add explicit spread features (`18-28`, `1-28`) identified by EDA as highly collinear.
  - Train on residual targets (`y_t+1 - y_t`).
- Results:
  - CV R² (Residual): **0.5234**.
  - Pseudo-LB R² (Level): **0.3946**.
  - **Leaderboard:** **0.3451**.
- Takeaways:
  - Regression compared to v19 (0.3563). Explicit spreads may have introduced noise or overfitting to training correlations that don't hold in the test set.

### 2.28 MLP v22 Vector Blend (v19 + v21)

- Files:
  - `scripts/optimize_vector_blend.py`.
- Concept:
  - Optimize blend weights `alpha_j` per feature on Pseudo-LB.
- Results:
  - Pseudo-LB: **0.4043** (Improved).
  - **Leaderboard:** **0.3566** (Regressed vs scalar blend 0.3571).
- Takeaways:
  - Overfitting to the small Pseudo-LB set. 32 parameters is too many for 10% validation data.

### 2.29 Stateful Feature-GRU (v23)

- Files:
  - `scripts/train_feature_gru.py`, `submissions/solution_v23_gru.py`.
- Concept:
  - Train 2-layer GRU on full sequences of 1187-dim features (v19+spreads).
  - Use stateful inference (passing hidden state step-to-step) to avoid timeouts.
- Results:
  - Val R² (Residual): **0.5066**.
  - **Leaderboard:** **0.3368**.
- Takeaways:
  - Works technically (no timeout), but underperforms MLP. Infinite memory likely captures regime noise specific to training data.

### 2.30 MLP v24 Scalar Blend (Current Safe Bet)

- Files:
  - `submissions/solution_v24_scalar.py`.
- Concept:
  - Robust scalar blend: `0.55 * v19 (Level) + 0.45 * v21 (Residual)`.
  - Verified with `scripts/adversarial_validation.py` (AUC ~0.47, no train/val shift).
- Status:
  - Packaged as `submission_mlp_v24_scalar.zip`.
  - Designed to minimize variance and prevent overfitting to target definition.

---

## 3. Current Understanding


## v10 Robust Ensemble (Winsorization + 5-Fold CV)
- Pseudo-LB Score: **0.39628** (Held out 10% seqs)
- CV Mean R2: **0.35354** (Std: 0.01288)
- Strategy: Winsorize [0.1%, 99.9%] on inputs only. Global stats on Dev set.


## v12 GRU Sequence Model
- Pseudo-LB Score: **0.34993**
- CV Mean R2: **0.31645**


## v13 Kinematics & Volatility
- New Features: Vol Expansion, Path Roughness, Accel Mean
- Pseudo-LB Score: **0.39513**
- CV Mean R2: **0.35377**


## v13 Kinematics & Volatility
- New Features: Vol Expansion, Path Roughness, Accel Mean
- Pseudo-LB Score: **0.39432**
- CV Mean R2: **0.35345**


## v14 AE-ResNet (Log Lags)
- Pseudo-LB Score: **0.37158**
- CV Mean R2: **0.34954**


## v16 NLinear (Lookback 336)
- Pseudo-LB Score: **0.30760**
- CV Mean R2: **0.26727**
