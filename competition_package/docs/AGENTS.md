# AGENT GUIDE – Wunder Challenge Project

This file tells an automated agent where to look for context first (docs and notes) and which code files are most relevant when working in this repo.

The idea: before doing anything non‑trivial, the agent should skim these files to understand the task, data, current solution, and past experiments.

---

## 1. High‑level Docs (read these first)

- `README.md`  
  - Competition statement and rules.  
  - Data format (`seq_ix`, `step_in_seq`, `need_prediction`, features `0`–`31`).  
  - Evaluation metric (R²) and streaming `PredictionModel` API.  
  - Submission format and structure.

- `docs/PLAN.md`  
  - Overall plan for the solution.  
  - Current status (what’s done, what’s next).  
  - Strategy: use Tsururu offline for exploration, then re‑implement in plain Python for submission.  
  - Iteration ideas for future work (lags, normalizers, model types).

- `datasets/DATA_DESCRIPTION.md`
  - Detailed description of `datasets/train.parquet`: shape, columns, dtypes.
  - Explanation of `need_prediction`, sequence lengths, and feature stats.
  - Mapping to Tsururu format (`id`, `date`, `value`) used in experiments.

- `datasets/DATASET_REPORT.md`
  - Comprehensive auto-generated markdown report of the full dataset.
  - Includes statistics, sample data, correlations, and distribution analysis.
  - **Use this to share dataset details with remote agents or collaborators.**

- `experiments/EXPERIMENT_LOG.md`  
  - Chronological log of experiments and results.  
  - Baselines (moving average), Tsururu CatBoost runs, custom lag‑MLP v1 and v2.  
  - Settings and outcomes (validation R², leaderboard scores, runtime).  
  - Future hypotheses and ideas to test.

- `docs/TEACHING_NOTES.md`
  - "Teacher notes" explaining the project step‑by‑step.
  - Math and reasoning behind the feature engineering, MLP training, and streaming inference.
  - How Tsururu fits as an offline exploration tool.
  - Good context for understanding *why* the current code looks the way it does.

- `docs/CATCH22_FEATURES_GUIDE.md`
  - Comprehensive student-friendly guide to catch22 time-series features.
  - Detailed explanations with examples, visualizations, and intuition for each feature.
  - Why catch22 is used offline only and how to design streaming-safe analogs.
  - Practice exercises and further reading suggestions.

- `docs/FEATURE_REGISTRY.md`
  - **Central definition** of all features used in the project (v1-v2x).
  - Maps feature names to formulas, motivations, and code implementations.
  - Tracks deprecated/failed features to avoid retrying bad ideas.

- `docs/FEATURE_GENERATION.md`
  - How to run catch22 offline labs and wire the outputs into supervised feature builders.
  - Guidance on designing streaming-safe analog features from heavy offline descriptors.

- `docs/SUBMISSION_GUIDE.md`
  - Step-by-step packaging instructions for MLP and CatBoost submissions.
  - Lists required files/artifacts to include in a ZIP and how to validate before upload.

- `docs/SOTA_IDEAS_STORE.md`
  - Parking lot of high-ROI modeling ideas, status, and next steps.
  - Use this to avoid repeating recent experiments and to pick the next candidate.

- `examples/simple/README.md`  
  - Description of the simple moving‑average example solution.  
  - Shows the minimal correct `PredictionModel` implementation.  
  - Useful for understanding the required streaming interface in a simpler setting.

---

## 2. Core Python Code (read next as needed)

- `src/utils.py` (re-exported via `utils.py` shim at repo root)  
  - Defines `DataPoint` and `ScorerStepByStep`.  
  - `ScorerStepByStep` implements the streaming evaluation logic used both locally and by the competition.  
  - Any changes to `PredictionModel` must respect this interface.

- `solution.py`  
  - Current submission model.  
  - Defines `PredictionModel` that:
    - Maintains per‑sequence state,  
    - Builds lag‑based features (raw lags + LastKnown‑delta + `step_in_seq/1000`),  
    - Applies normalization and feeds a small MLP.  
  - Contains a `__main__` block to evaluate on `train.parquet` using `ScorerStepByStep`.

- `scripts/train_model.py`  
  - Offline trainer and feature builder for the MLP.  
  - **v10 Update**: Implements **5-Fold Cross-Validation** and **Pseudo-LB** splitting.
  - Builds supervised `(X, y)` with **Winsorization** (input clipping to [0.1%, 99.9%] quantiles).
  - Trains 5 independent models (one per fold) and saves them as `models/lag_mlp_fold*.pth`.
  - Computes global normalization/clipping stats on the Dev set only.

- `scripts/leakage_check.py`  
  - Verifies that `build_supervised_dataset` does not leak future information:  
    - Checks train/val sequence disjointness,  
    - Reconstructs lag windows and targets from raw data for random samples.

- `scripts/tsururu_experiment.py`  
  - Offline Tsururu experiments (univariate CatBoost on feature `0`, plus some exploratory multivariate attempts).  
  - Not used in submission; used only for learning and guiding feature design.

- `scripts/train_catboost_experiment.py`  
  - Offline CatBoost baseline / feature lab using the same lag+delta features as `scripts/train_model.py`.  
  - Trains a single multi-output `CatBoostRegressor(loss_function="MultiRMSE")` and reports validation R².  
  - Used to benchmark feature sets (lags vs lags+delta vs future additions) without depending on Tsururu.

- `scripts/compute_catch22_features.py`  
  - Offline automated feature generation script (planned lab tool).  
  - Computes per-sequence, per-dimension catch22 features (22 canonical time-series stats) and saves them (e.g. to `datasets/catch22_per_seq.npz`) for later use in `scripts/train_model.py` / `solution.py`.  
  - Not used in the submission directly; its outputs are small numeric feature tables that can be safely loaded at runtime.

- `examples/simple/solution.py`  
  - Minimal working solution using a moving‑average strategy.  
  - Good reference for the simplest correct `PredictionModel`.

---

## 3. Agent Instructions / Expectations

When working on this repo, an agent should:

1. **Start with docs**:
   - Read `README.md`, `docs/PLAN.md`, `datasets/DATA_DESCRIPTION.md`, `experiments/EXPERIMENT_LOG.md`, `docs/TEACHING_NOTES.md`, and `examples/simple/README.md` to understand:
     - The data and task,  
     - The current solution path,  
     - What has already been tried and how it performed.

2. **Respect the competition interface**:
   - `solution.py` must define `PredictionModel` with `predict(self, data_point: DataPoint) -> np.ndarray | None`.  
   - Must return `None` when `need_prediction == 0`, and a 32‑dim numpy array when `need_prediction == 1`.  
   - Must handle `seq_ix` changes by resetting internal state.

3. **Two model roles: lab vs submission**:
   - The primary **submission model** is a small lag-based MLP:
     - Trained via `scripts/train_model.py` on lag+delta+rolling features (and, in future, optionally augmented with precomputed automated features like catch22).  
     - Implemented in `solution.py` as a streaming `PredictionModel`, using only allowed libraries (NumPy, pandas, PyTorch).  
   - CatBoost models are used **offline only**:
     - `scripts/train_catboost_experiment.py` trains a `CatBoostRegressor(loss_function="MultiRMSE")` on the exact same supervised `(X, y)` features that the MLP uses.  
     - CatBoost is treated as a strong baseline / oracle for feature sets, not as the runtime submission model (unless explicitly decided later and allowed by the environment).

3. **Preserve leak‑free supervision**:
   - Any changes to how `(X, y)` are built in `scripts/train_model.py` must continue to:
     - Use only past information to predict the next step,  
     - Split by `seq_ix` for train/val,  
     - Compute normalization on train only.  
   - If changing the feature spec, update both `scripts/train_model.py` and `solution.py` consistently.

4. **Keep runtime constraints in mind**:
   - Scoring over full `train.parquet` currently takes ~15 seconds locally.  
   - The competition gives 60 minutes on CPU for the full test set; any new model must remain comfortably below this.

5. **Use Tsururu only for offline exploration**:
   - Tsururu is not available in the scoring environment; do not introduce a runtime dependency on it.  
   - It can be used to explore lag/normalizer/model ideas, but final implementations should be in plain Python (NumPy / PyTorch / possibly CatBoost).

6. **Use automated feature generation as an offline lab**:
   - Tools like **catch22** (via `compute_catch22_features.py`) should be used offline to generate small, precomputed feature tables (e.g. per-sequence descriptors).  
   - These precomputed features can then be:
     - Loaded in `scripts/train_model.py` and appended to the supervised feature vector `X` for both MLP and CatBoost experiments,  
     - Loaded in `solution.py` and concatenated with lag-based features at inference time (no heavy library imports in submission code).  
   - The MLP serves as the **fast feature lab** (quick retrains to test if new features help), while CatBoost is a **slow, strong oracle** that is run only occasionally on the best feature sets.

---

## 4. Suggested Workflow for Future Agents

When asked to improve or modify the solution:

1. Skim the docs listed above to recall current models and scores.  
2. Check `experiments/EXPERIMENT_LOG.md` to avoid duplicating old experiments.  
3. If changing features or the model:
   - Update `scripts/train_model.py` (offline training) and retrain.  
   - Mirror the same feature logic inside `PredictionModel` in `solution.py`.  
   - Run `python solution.py` to ensure streaming behavior and runtime are acceptable.  
4. If adding a new experiment:
   - Append a short, clear entry to `experiments/EXPERIMENT_LOG.md` with settings and results.  
   - Update `docs/PLAN.md` or `docs/TEACHING_NOTES.md` if the change is conceptual or structural.  

---

## 5. Environment / Tooling Notes

- Prefer using the project virtual environment at:
  - `/Users/sergei/PycharmProjects/WunderSex/.venv`
- When running training or scoring scripts inside `competition_package`, use:
  - `../.venv/bin/python scripts/train_model.py`
  - `../.venv/bin/python solution.py`
- Keep changes compatible with a CPU-only PyTorch setup.

---

## 6. Validation Strategy Notes

- Default training split in `scripts/train_model.py` is **80/20 by `seq_ix`**, which is good but still optimistic relative to the hidden leaderboard.  
- For more reliable estimates before submitting:
  - Use **K-fold CV by `seq_ix`** (e.g. 5 folds): shuffle sequence IDs once, then rotate which fold is used as validation; report mean/std of validation R².  
  - For streaming behavior, run `solution.py` on a **held-out subset of sequences** (e.g. 10–20% of `seq_ix`) and treat that as a local “pseudo-leaderboard” instead of just using R² on the full train file.  
- Recent experience (v8 residual-target ensemble) showed that improvements in train-file streaming R² do **not** always translate to better leaderboard scores; prefer sequence-level CV and held-out seq splits as decision criteria for new models.

This structure should give any future agent enough context to make informed changes without breaking the competition contract or re‑doing work that’s already been done.
