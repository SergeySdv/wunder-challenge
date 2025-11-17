# Wunder Challenge – Solution Plan ✅/⬜

High‑level goal:  
Use **Tsururu** locally on `train.parquet` to discover strong forecasting strategies (lags, normalizers, model type, hyperparams), then **re‑implement the winning setup** as a deterministic, streaming `PredictionModel` that does **not** depend on Tsururu at runtime.

## 0. Current Status

- [x] Download starter pack and inspect structure  
- [x] Run basic EDA on `datasets/train.parquet`  
- [x] Implement simple moving‑average baseline in `competition_package/solution.py`  
- [x] Establish strong baseline using Tsururu offline (univariate CatBoost + lags)  
- [x] Implement first optimized streaming model for submission (lag‑MLP)  

## 1. Offline Exploration with Tsururu (Local Only)

- [x] Install Tsururu and extra deps in local env (no impact on submission image)  
  - [x] `pip install -U tsururu[catboost]` (plus `torch`)  
- [x] Load `competition_package/datasets/train.parquet` into a DataFrame  
- [x] Map competition data to Tsururu format:
  - [x] Treat each `seq_ix` as a separate time series (`id`)  
  - [x] Create synthetic `date` column from `step_in_seq` (Day offset)  
  - [ ] Extend to multivariate targets using all feature columns (`0`–`31`)  
- [x] Define a **TSDataset** and **Pipeline** (current best config):
  - [x] Use `Pipeline.easy_setup` with `target_lags=10`, `date_lags=1`  
  - [x] Use `target_normalizer='standard_scaler'` (no delta/ratio regime)  
- [x] Try strategy with **horizon = 1**:
  - [x] Recursive (global, univariate on feature `0`)  
  - [ ] Direct / MIMO / FlatWideMIMO if needed later  
- [x] Evaluate CatBoost model:
  - [x] Model: CatBoostRegressor (loss `RMSE`, depth 6, 500 iters, early stopping)  
  - [x] Validation: `KFoldCrossValidator(n_splits=3)` inside `MLTrainer`  
  - [x] Achieved CV score ≈ **R² ~ 0.9325 ± 0.0012** on feature `0`  
  - [x] Fit time ≈ **16.6 s**, forecast time ≈ **1.2 s** on train set  
- [ ] (Optional) Explore:
  - [ ] LastKnownNormalizer / DifferenceNormalizer instead of standard scaler  
  - [ ] Larger/smaller `history` windows and different lag sets  
  - [ ] NN models (DLinear, etc.) for comparison  
- [ ] Select a **final configuration** to mirror in custom code based on:
  - [ ] Validation R² across all 32 features  
  - [ ] Feature‑engineering pattern (lags, normalizers, stats)  
  - [ ] Inference cost estimate (must be CPU‑fast and simple to implement)  

## 2. Custom Streaming Model (Competition‑Ready)

Re‑implement the selected Tsururu strategy using only allowed core libs (e.g., NumPy, pandas, scikit‑learn, CatBoost / PyTorch if confirmed available).

- [x] Design streaming feature‑engineering to mirror best Tsururu pipeline:
  - [x] Maintain per‑sequence history buffer inside `PredictionModel`  
  - [x] Implement lag features (last 10 steps of all 32 features)  
  - [x] Optionally add step position feature (implemented as `step_in_seq/1000`)  
- [x] Implement an **offline training script** (`train_model.py`):
  - [x] Read `train.parquet`  
  - [x] Build the same features **batch‑wise** that `PredictionModel` builds online  
  - [x] Split by `seq_ix` into train/validation (80% / 20%, sequence‑disjoint)  
  - [x] Train the chosen model:
    - [ ] If gradient boosting: one model per feature (32 regressors)  
    - [x] If NN: a small CPU‑friendly network (single‑hidden‑layer MLP)  
  - [x] Save weights to disk (`models/lag_mlp.pth` and `models/lag_mlp_normalization.npz`)  
- [x] Update `competition_package/solution.py`:
  - [x] Load saved weights in `PredictionModel.__init__`  
  - [x] Manage per‑sequence state and history (reset on new `seq_ix`)  
  - [x] Build features **incrementally** at each `predict()` call  
  - [x] Return `None` when `need_prediction == 0`  
  - [x] Return a `(32,)` NumPy array when `need_prediction == 1`  
  - [x] Enforce determinism (fixed random seeds, single‑thread if needed)  

## 3. Performance & Correctness Checks

- [x] Local correctness:
  - [x] Use `ScorerStepByStep` on `train.parquet` with new `PredictionModel`  
  - [x] Verify shape, `None` handling, and sequence resets are correct  
- [x] Runtime profiling (very important):
  - [x] Measure runtime on full `train.parquet` (≈ 12–15 seconds locally)  
  - [x] Extrapolate to full test size (~500k+ rows) to stay well under 60 minutes  
  - [ ] If needed, simplify model / reduce feature count until safe  
- [ ] Determinism:
  - [ ] Run evaluation twice and confirm identical predictions and R²  

## 4. Packaging & Submission

- [ ] Prepare submission folder (minimal contents):
  - [ ] `solution.py` (entry point with `PredictionModel`)  
  - [ ] Model weight files (e.g., `cat_0.cbm`–`cat_31.cbm` or `model.pth`)  
  - [ ] Any small helper modules strictly needed at inference  
- [ ] Verify no Tsururu dependence in submission:
  - [ ] `solution.py` and helpers import only allowed libs (NumPy/pandas/sklearn/torch/etc.)  
  - [ ] No `import tsururu` in submission code  
- [ ] Create archive from the solution directory:
  - [ ] `zip -r ../submission.zip .`  
- [ ] Test the zip locally if there is a provided scorer wrapper (optional)  
- [ ] Submit and monitor leaderboard performance  

## 5. Iteration Ideas (Optional)

### 5.1 Ideas from Tsururu (what we already learned)

- [x] Lag features are the core:
  - [x] Use last `K` points (history window) to build features.  
  - [x] Train on the “next” point as the supervised target.  
  - [x] Global models over many series (one model across all `seq_ix`) work well.  
- [ ] Normalizers beyond standard scaling:
  - [ ] Difference‑style features: deltas or ratios vs previous value.  
  - [ ] LastKnown‑style features: deltas or ratios vs the **last lag** (shape of the window).  
- [x] Modeling style:
  - [x] Use one global multi‑output model (32‑dim outputs) rather than local per‑sequence models.  
  - [x] Do not concatenate different sequences at the same time step (sequences are independent).  

### 5.2 Next concrete iteration – Lag + LastKnown‑delta features

- [x] Extend `train_model.py` feature builder:
  - [x] Keep existing raw lag window: `[state(t‑9), …, state(t)]` flattened.  
  - [x] Add **LastKnown‑delta** features per dimension:  
    - For each lag window `[x_{t‑9}, …, x_t]`, compute deltas `x_{t-k} − x_t`.  
  - [x] Add simple **rolling stats** over the lag window (per feature): mean and std over the last 10 steps.  
  - [x] Keep `step_in_seq / 1000` as a simple position feature.  
- [x] Update normalization and storage:
  - [x] Recompute `x_mean`, `x_std` on the **expanded** feature vector.  
  - [x] Save updated normalization to `models/lag_mlp_normalization.npz` (documenting new input_dim).  
- [x] Retrain MLP with new features:
  - [x] Keep the same architecture (single hidden layer, 64 units) and `N_EPOCHS=10` for stability.  
  - [x] Track validation mean R²; compare to previous baselines (~0.42 → ~0.428).  
- [x] Integrate into `solution.py`:
  - [x] Mirror the new feature computation (raw lags + LastKnown‑delta + rolling stats + position).  
  - [x] Ensure the feature order matches `train_model.py`.  
  - [x] Keep the same streaming interface (`state_history`, `step_in_seq/1000`).  
- [ ] Evaluate:
  - [x] Run `python solution.py` to get new train‑file R² and check runtime.  
  - [ ] Package and submit this version if leaderboard improves and runtime remains safe.  

### 5.3 Longer‑term options

- [ ] Use Tsururu for additional **univariate** experiments:
  - [ ] Try different `target_lags` and normalizers (Difference / LastKnown) on a few features (e.g., 0, 5, 10) to see which patterns generalize.  
- [ ] Mirror any clearly better lag/normalizer patterns in `train_model.py`.  
- [ ] Optionally train CatBoost models on our own lag + delta features (one model per feature) if CatBoost is confirmed available in the scoring env.  
- [ ] Add lightweight ensembling (e.g., blend CatBoost + MLP) if runtime allows.  
- [ ] Tune hyperparameters (lags, hidden sizes, epochs) for best validation R² under time constraints.  

### 5.4 Automated Feature Generation Lab (catch22, etc.)

Goal: use automated time‑series feature extractors (especially **catch22**) as an **offline lab** to design better feature sets, then re‑implement the winning ideas in our simple lag+MLP pipeline.

- [ ] Add an offline feature extraction script (e.g., `compute_catch22_features.py`):
  - [ ] For each `seq_ix` and each feature dimension (0–31), compute the 22 catch22 features on the full 1000‑step series.  
  - [ ] Concatenate across 32 dims → 704 features per sequence.  
  - [ ] Save as `datasets/catch22_per_seq.npz` with `seq_ids` and `catch22_features` (shape `(n_seqs, 704)`).  
- [ ] Augment supervised features in `train_model.py`:
  - [ ] Load `catch22_per_seq.npz` once at startup.  
  - [ ] When building each sample `(X_t, y_t)`, look up the precomputed 704‑dim catch22 vector for its `seq_ix` and append it to the lag+delta+stats+step feature vector.  
  - [ ] Retrain the MLP and compare validation mean R² to the current v3 baseline (~0.428).  
- [ ] Mirror catch22 features in `solution.py`:
  - [ ] Bundle `catch22_per_seq.npz` in the submission and load it in `PredictionModel.__init__`.  
  - [ ] In `_build_features`, after building lag+delta+rolling+step features, append the per‑sequence catch22 vector (no runtime call to catch22 itself).  
- [ ] Feature validation workflow (MLP as fast lab):
  - [ ] Fix the MLP architecture and training hyperparameters (hidden size, epochs) for stability.  
  - [ ] For each new feature set (e.g., +catch22, +trend, +longer‑window stats):  
    - [ ] Retrain MLP once, record validation mean R² from `train_model.py`,  
    - [ ] Run `python solution.py` to record streaming train‑file R² and runtime,  
    - [ ] Log results in `EXPERIMENT_LOG.md` (config + scores).  
  - [ ] Keep only feature sets that provide a meaningful lift (e.g., +0.01 R² or more) without hurting runtime.  
- [ ] Final feature scoring & selection for submission:
  - [ ] Once a few strong feature sets are identified via MLP, optionally run CatBoost MultiRMSE offline on the best one or two as a high‑cost “oracle” check.  
  - [ ] Use model performance (val R² and streaming R²) and, if needed, CatBoost feature importance to decide which features to keep.  
  - [ ] Lock in the chosen feature set and MLP configuration for the final submission, avoiding further structural changes close to the deadline.  
