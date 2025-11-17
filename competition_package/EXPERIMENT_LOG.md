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

---

## 3. Current Understanding

- The data is already roughly standardized; simple lag features are powerful.  
- A compact global model (shared across sequences) works well: no need for one model per `seq_ix`.  
- Univariate Tsururu+CatBoost with good lags/normalization can reach very high R² (~0.93) on a single feature.  
- Our custom multivariate lag‑MLP:
  - v1 (raw lags) achieves ~0.41 val R² offline and ~0.3266 leaderboard R².  
  - v2 (lags + LastKnown‑delta) achieves ~0.42+ val R² offline and ~0.3293 leaderboard R².  
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
