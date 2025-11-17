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

- `PLAN.md`  
  - Overall plan for the solution.  
  - Current status (what’s done, what’s next).  
  - Strategy: use Tsururu offline for exploration, then re‑implement in plain Python for submission.  
  - Iteration ideas for future work (lags, normalizers, model types).

- `datasets/DATA_DESCRIPTION.md`  
  - Detailed description of `datasets/train.parquet`: shape, columns, dtypes.  
  - Explanation of `need_prediction`, sequence lengths, and feature stats.  
  - Mapping to Tsururu format (`id`, `date`, `value`) used in experiments.

- `EXPERIMENT_LOG.md`  
  - Chronological log of experiments and results.  
  - Baselines (moving average), Tsururu CatBoost runs, custom lag‑MLP v1 and v2.  
  - Settings and outcomes (validation R², leaderboard scores, runtime).  
  - Future hypotheses and ideas to test.

- `TEACHING_NOTES.md`  
  - “Teacher notes” explaining the project step‑by‑step.  
  - Math and reasoning behind the feature engineering, MLP training, and streaming inference.  
  - How Tsururu fits as an offline exploration tool.  
  - Good context for understanding *why* the current code looks the way it does.

- `examples/simple/README.md`  
  - Description of the simple moving‑average example solution.  
  - Shows the minimal correct `PredictionModel` implementation.  
  - Useful for understanding the required streaming interface in a simpler setting.

---

## 2. Core Python Code (read next as needed)

- `utils.py`  
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

- `train_model.py`  
  - Offline trainer and feature builder for the MLP.  
  - Builds supervised `(X, y)`:
    - X: last 10 states for all 32 features, plus LastKnown‑delta features and a position feature.  
    - y: next 32‑dim state.  
  - Handles train/validation split by `seq_ix`, normalization, MLP training, and saving weights/normalization to `models/`.

- `leakage_check.py`  
  - Verifies that `build_supervised_dataset` does not leak future information:  
    - Checks train/val sequence disjointness,  
    - Reconstructs lag windows and targets from raw data for random samples.

- `tsururu_experiment.py`  
  - Offline Tsururu experiments (univariate CatBoost on feature `0`, plus some exploratory multivariate attempts).  
  - Not used in submission; used only for learning and guiding feature design.

- `train_catboost_experiment.py`  
  - Offline CatBoost baseline / feature lab using the same lag+delta features as `train_model.py`.  
  - Trains a single multi-output `CatBoostRegressor(loss_function="MultiRMSE")` and reports validation R².  
  - Used to benchmark feature sets (lags vs lags+delta vs future additions) without depending on Tsururu.

- `compute_catch22_features.py`  
  - Offline automated feature generation script (planned lab tool).  
  - Computes per-sequence, per-dimension catch22 features (22 canonical time-series stats) and saves them (e.g. to `datasets/catch22_per_seq.npz`) for later use in `train_model.py` / `solution.py`.  
  - Not used in the submission directly; its outputs are small numeric feature tables that can be safely loaded at runtime.

- `examples/simple/solution.py`  
  - Minimal working solution using a moving‑average strategy.  
  - Good reference for the simplest correct `PredictionModel`.

---

## 3. Agent Instructions / Expectations

When working on this repo, an agent should:

1. **Start with docs**:
   - Read `README.md`, `PLAN.md`, `DATA_DESCRIPTION.md`, `EXPERIMENT_LOG.md`, `TEACHING_NOTES.md`, and `examples/simple/README.md` to understand:
     - The data and task,  
     - The current solution path,  
     - What has already been tried and how it performed.

2. **Respect the competition interface**:
   - `solution.py` must define `PredictionModel` with `predict(self, data_point: DataPoint) -> np.ndarray | None`.  
   - Must return `None` when `need_prediction == 0`, and a 32‑dim numpy array when `need_prediction == 1`.  
   - Must handle `seq_ix` changes by resetting internal state.

3. **Two model roles: lab vs submission**:
   - The primary **submission model** is a small lag-based MLP:
     - Trained via `train_model.py` on lag+delta+rolling features (and, in future, optionally augmented with precomputed automated features like catch22).  
     - Implemented in `solution.py` as a streaming `PredictionModel`, using only allowed libraries (NumPy, pandas, PyTorch).  
   - CatBoost models are used **offline only**:
     - `train_catboost_experiment.py` trains a `CatBoostRegressor(loss_function="MultiRMSE")` on the exact same supervised `(X, y)` features that the MLP uses.  
     - CatBoost is treated as a strong baseline / oracle for feature sets, not as the runtime submission model (unless explicitly decided later and allowed by the environment).

3. **Preserve leak‑free supervision**:
   - Any changes to how `(X, y)` are built in `train_model.py` must continue to:
     - Use only past information to predict the next step,  
     - Split by `seq_ix` for train/val,  
     - Compute normalization on train only.  
   - If changing the feature spec, update both `train_model.py` and `solution.py` consistently.

4. **Keep runtime constraints in mind**:
   - Scoring over full `train.parquet` currently takes ~15 seconds locally.  
   - The competition gives 60 minutes on CPU for the full test set; any new model must remain comfortably below this.

5. **Use Tsururu only for offline exploration**:
   - Tsururu is not available in the scoring environment; do not introduce a runtime dependency on it.  
   - It can be used to explore lag/normalizer/model ideas, but final implementations should be in plain Python (NumPy / PyTorch / possibly CatBoost).

6. **Use automated feature generation as an offline lab**:
   - Tools like **catch22** (via `compute_catch22_features.py`) should be used offline to generate small, precomputed feature tables (e.g. per-sequence descriptors).  
   - These precomputed features can then be:
     - Loaded in `train_model.py` and appended to the supervised feature vector `X` for both MLP and CatBoost experiments,  
     - Loaded in `solution.py` and concatenated with lag-based features at inference time (no heavy library imports in submission code).  
   - The MLP serves as the **fast feature lab** (quick retrains to test if new features help), while CatBoost is a **slow, strong oracle** that is run only occasionally on the best feature sets.

---

## 4. Suggested Workflow for Future Agents

When asked to improve or modify the solution:

1. Skim the docs listed above to recall current models and scores.  
2. Check `EXPERIMENT_LOG.md` to avoid duplicating old experiments.  
3. If changing features or the model:
   - Update `train_model.py` (offline training) and retrain.  
   - Mirror the same feature logic inside `PredictionModel` in `solution.py`.  
   - Run `python solution.py` to ensure streaming behavior and runtime are acceptable.  
4. If adding a new experiment:
   - Append a short, clear entry to `EXPERIMENT_LOG.md` with settings and results.  
   - Update `PLAN.md` or `TEACHING_NOTES.md` if the change is conceptual or structural.  

This structure should give any future agent enough context to make informed changes without breaking the competition contract or re‑doing work that’s already been done.
