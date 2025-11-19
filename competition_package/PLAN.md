# Wunder Challenge – Solution Plan ✅/⬜

High‑level goal:  
Use **Tsururu** locally on `train.parquet` to discover strong forecasting strategies (lags, normalizers, model type, hyperparams), then **re‑implement the winning setup** as a deterministic, streaming `PredictionModel` that does **not** depend on Tsururu at runtime.

## 0. Current Status

- [x] Download starter pack and inspect structure  
- [x] Run basic EDA on `datasets/train.parquet`  
- [x] Implement simple moving‑average baseline in `competition_package/solution.py`  
- [x] Establish strong baseline using Tsururu offline (univariate CatBoost + lags)  
- [x] Implement first optimized streaming model for submission (lag‑MLP)  
- [x] Iterate feature sets and submissions:
  - [x] v1 – raw lags only → LB ~0.3266.  
  - [x] v2 – lags + LastKnown‑delta → LB ~0.3293.  
  - [x] v3 – lags + LastKnown‑delta + rolling mean/std → small offline/streaming gain.  
  - [x] v4 – v3 + per‑sequence catch22 (lab only; overfit LB ~0.15 when used in submission).  
  - [x] v5 – v3 + streaming‑safe analogs (lag‑1 autocorr + persistence fraction) → LB ~**0.3390** with safe runtime.  
  - [x] v6 – v5 + extra short‑window autocorr (lags 2–3), robust window stats, and per‑feature trend → LB ~**0.3400**; strong streaming R² on `train.parquet`.  
  - [x] v7 – deeper funnel MLP + LR scheduler + 3‑seed ensemble on level targets → LB ~**0.3469** (current best submission); streaming train R² ≈ **0.4488**.  
  - [x] v8 – same v7 features/architecture but with residual targets `state(t+1) − state(t)` and ensemble averaging of deltas → streaming train R² ≈ **0.4529**, but LB dropped to **~0.3378**; kept as a lab experiment only (code reverted to v7‑style level targets for future training).  
  - [x] v9 – v7 feature set + additional spread features between several highly correlated pairs (e.g. 18–28, 11–30, 0–21, 7–31, 1–28, 3–4); streaming train R² improved further to ≈ **0.4535**, but the leaderboard score was **~0.3461** (slightly below v7), so this is kept as a lab‑only idea and the main code has been reverted to the simpler v7 ensemble.  

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

- [x] Prepare submission folder (minimal contents):
  - [x] `solution.py` (entry point with `PredictionModel`)  
  - [x] Model weight files (e.g., `lag_mlp.pth`, `lag_mlp_normalization.npz`).  
  - [x] Any small helper modules strictly needed at inference (`utils.py`).  
- [x] Verify no Tsururu dependence in submission:
  - [x] `solution.py` and helpers import only allowed libs (NumPy/pandas/sklearn/torch/etc.).  
  - [x] No `import tsururu` in submission code.  
- [x] Create archives from the solution directory:
  - [x] `submission.zip`, `submission_lag_delta.zip`, `submission_mlp_catch22.zip`, `submission_mlp_v3.zip`, `submission_mlp_v5_streaming_analogs.zip`.  
- [x] Test zips locally where practical (via `python solution.py` on `train.parquet`).  
- [x] Submit and monitor leaderboard performance (currently best: v5 streaming‑analog MLP at ~0.3390).  
  - As of v7: best leaderboard score is **~0.3469** from the v7 level‑target 3‑seed ensemble.  
  - v8 residual‑target ensemble improved train streaming R² but **hurt** LB (~0.3378), suggesting some over‑optimization to the train distribution; future work should be validated via sequence‑level cross‑validation and held‑out seq splits, not train‑file R² alone.  

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

Goal: use automated time‑series feature extractors (especially **catch22**) as an **offline lab** to design better feature sets, then re‑implement the winning ideas in our simple lag+MLP pipeline using streaming‑safe analogs.

- [x] Add an offline feature extraction script (`compute_catch22_features.py`):
  - [x] For each `seq_ix` and each feature dimension (0–31), compute the 22 catch22 features on the full 1000‑step series.  
  - [x] Concatenate across 32 dims → 704 features per sequence.  
  - [x] Save as `datasets/catch22_per_seq.npz` with `seq_ids` and `catch22_values` (shape `(n_seqs, 32, 22)`).  
- [x] Augment supervised features in `train_model.py` for offline labs:
  - [x] Load `catch22_per_seq.npz` once at startup.  
  - [x] When building each sample `(X_t, y_t)`, look up the precomputed 704‑dim catch22 vector for its `seq_ix` and append it to the lag+delta+stats+step feature vector when `use_catch22=True`.  
  - [x] Retrain the MLP and compare validation mean R² to the current v3 baseline (~0.428); v4 (with catch22) reaches ~0.4335 offline but does not generalize when used directly in submissions.  
- [x] Analyze catch22 contributions and design streaming‑safe analogs:
  - [x] Use CatBoost (`catch22_feature_importance.py`) to compute the global importance split between v3 features and the catch22 block (~88% vs ~12%), and identify the most important catch22 statistics (spectral energy/centroid, autocorrelation time, persistence, local trend).  
  - [x] From these findings, define a small set of cheap, streaming‑friendly features (short‑window lag‑1 autocorrelation and a simple persistence fraction) and test them as extensions of the v3 feature set (v5 experiment).  
- [x] Submission‑side usage (updated plan):
  - [x] Do **not** mirror full per‑sequence catch22 vectors into `solution.py` (this was tried and caused strong overfitting and a public leaderboard drop).  
  - [x] Instead, implement only streaming‑safe analogs in both `train_model.py` and `solution.py`, computed from the rolling `n_lags` buffer and current `step_in_seq` (v5/v6 now in `solution.py`).  
- [x] Feature validation workflow (MLP as fast lab):
  - [x] Fix the MLP architecture and training hyperparameters (hidden size, epochs) for stability (64 hidden units, 10 epochs).  
  - [x] For the new v5 feature set (+lag‑1 autocorr, +persistence fraction):  
    - [x] Retrain MLP once, record validation mean R² from `train_model.py` (~0.4318).  
    - [x] Run `python solution.py` to record streaming train‑file R² and runtime (~0.36+ R², ~35–40 s).  
    - [x] Log results in `EXPERIMENT_LOG.md` (config + scores).  
  - [x] Keep only feature sets that provide a meaningful lift (e.g., +0.01 R² or more) without hurting runtime or generalization (v5/v6 currently pass this check vs v1–v3).  
- [x] Final feature scoring & selection for submission:
  - [x] Once a few strong feature sets are identified via MLP, optionally run CatBoost MultiRMSE offline on the best one or two as a high‑cost “oracle” check (done for v5).  
  - [x] Use model performance (val R² and streaming R²) and, if needed, CatBoost feature importance to decide which features to keep (we kept the v6 analog extensions, not per‑sequence catch22).  
  - [x] Lock in the chosen feature set and MLP configuration for the current submission (v6 features + 1‑hidden‑layer MLP), while treating further changes as optional, incremental experiments.  

### 5.5 Future Score‑Improvement Ideas (Post‑v5)

If we want to push beyond the current ~0.339 leaderboard score, possible next experiments include:

- [x] **v10 – Robustness & Strict Validation (Winsorization + CV):**
  - Implemented "autofin"-style outlier handling: clip inputs to [0.1%, 99.9%] quantiles.
  - Switched to **5-Fold CV** by sequence + **Pseudo-LB** (10% held out) to better estimate generalization.
  - Result: CV R² ~0.354, Pseudo-LB R² ~0.389. Packaged in `submission_mlp_v10_robust_ensemble.zip`.
  - **Leaderboard Score:** **0.3513** (New Best vs previous 0.3469).

- **Model capacity tweaks (same v5 features):**
  - Try hidden size 128 or a shallow 2‑layer MLP (e.g., 128 → 64 → 32) with the v5 feature set.  
  - Gradually increase epochs (e.g., 15–20) with early stopping on validation R².  
- **v6 – richer short‑window statistics (building on v5):**
  - Extend the current streaming‑safe analog set beyond `ac_lag1` and persistence fraction, still using only the last 10 steps:  
    - Add short‑window autocorrelation at lags 2 and 3, plus a simple aggregate like `sum(|acf_1..3|)` per feature.  
    - Add robust rolling statistics per feature over the 10‑step window: quantiles (25/50/75), IQR, skewness, kurtosis, coefficient of variation.  
    - Add a tiny local trend block per feature: slope of a least‑squares fit over the last 10 points, simple `R²` of that fit, and a crude curvature indicator (difference between early‑half and late‑half slopes).  
  - Implement these in `train_model.py` first, retrain the MLP, and log offline/streaming R² as “v6” in `EXPERIMENT_LOG.md`.  
  - Once offline gains are confirmed, mirror the exact same computations into `solution.py` to keep submission streaming‑safe.  
- **Multi‑scale lags (if runtime allows and v6 helps):**
  - Explore adding a few coarser lags (e.g., at offsets 2, 5, 20) on top of the existing 10‑step window, guided by Tsururu results.  
- **Heavier models (later phase):**
  - Small GRU/LSTM per sequence, or a slightly larger CatBoost MultiRMSE, as long as runtime on train/test stays safe.  
  - Consider simple ensembles (e.g., average MLP and CatBoost predictions) if file size and CPU budget allow.  

All of these should follow the same pattern as v5: prototype offline, verify streaming consistency, measure train/val R², and then test a single clean submission.  

### 5.6 Ideas Inspired by External Finance Tooling (e.g. `autofin`)

We do **not** plan to import or depend on external libraries like `autofin` inside the submission; instead we can borrow a few design ideas and, if they prove useful, re‑implement them directly in our own codebase:

- **Robustness to outliers (winsorization / clipping):**
  - Investigate per‑feature clipping of states or short‑horizon returns/deltas based on train‑only quantiles (e.g., 0.05% / 99.95% or 0.1% / 99.9%), to reduce the influence of rare spikes without changing the bulk distribution.  
  - If experiments show this helps validation/held‑out R², incorporate it as a simple preprocessing step applied consistently in both `train_model.py` and `solution.py` (next‑step “v10”‑style idea).  

- **Stronger time‑series validation schemes:**
  - Treat `seq_ix` as a “group” and build K‑fold cross‑validation over sequences with fixed folds and, optionally, pseudo‑leaderboard folds that remain untouched until late in iteration (mirroring `group_time_series` splitting ideas).  
  - Consider simple rolling/expanding splits over `seq_ix` indices (earlier seqs as train, later as validation) as an additional robustness check.  
  - Use these CV schemes to decide whether new feature ideas (like spreads, clipping, etc.) actually generalize, not just improve train‑file R².  

- **Return/volatility perspective (already partially explored):**
  - The “predict returns instead of prices” idea corresponds to our v8 residual‑target ensemble; given its worse leaderboard score, we treat this as a cautionary example rather than the main direction.  
  - Any further ideas in this vein (e.g., volatility‑scaled targets) should be tested very carefully on strong held‑out seq splits before being considered for submission.  
