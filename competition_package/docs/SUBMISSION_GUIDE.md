# Submission Guide – MLP and CatBoost Variants

This guide describes how to train, evaluate, and package submission ZIPs for both the **MLP-based** and **CatBoost-based** solutions in this project.

All paths below assume you are in the project root:

```bash
cd /Users/sergei/PycharmProjects/WunderSex
```

and the competition code lives in:

```bash
cd competition_package
```

---

## 1. Environment Setup

Create and activate a Python environment with the required packages (NumPy, pandas, PyTorch, CatBoost, pyarrow, etc.).

Example (adjust to your setup):

```bash
cd /Users/sergei/PycharmProjects/WunderSex
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install numpy pandas pyarrow torch catboost tqdm
```

Make sure the competition data file exists:

```bash
ls competition_package/datasets/train.parquet
```

---

## 2. MLP-Based Submission (lag-MLP, v6 features)

The MLP submission is implemented in `competition_package/solution.py`. It:

- Maintains a rolling window of the last `n_lags` states per sequence.  
- Builds lag-based + short-window statistics + trend features from that buffer.  
- Applies a small MLP trained offline in `train_model.py`.  

### 2.1 Train the MLP

From `competition_package`:

```bash
cd competition_package
python scripts/train_model.py
```

This will:

- Load `datasets/train.parquet`.  
- Split by `seq_ix` into train/validation (80%/20%, sequence-disjoint).  
- Build supervised `(X, y)` using the current feature set (v6).  
- Standardize `X` using train statistics.  
- Train the `LagMLP` model.  
- Save:
  - `models/lag_mlp.pth` – MLP weights and metadata.  
  - `models/lag_mlp_normalization.npz` – `x_mean`, `x_std`, `n_lags`, `feature_cols`.  

### 2.2 Optional: Check Offline vs Streaming Features

To confirm that offline feature building matches the streaming implementation in `solution.py`:

```bash
cd competition_package
python scripts/feature_consistency_check.py
```

This will sample a few points, compare normalized features from:

- `build_supervised_dataset(...)` (offline), and  
- `PredictionModel._build_features(...)` (online),  

and report any mismatches.

### 2.3 Evaluate Streaming MLP on `train.parquet`

From `competition_package`:

```bash
python solution.py
```

This will:

- Run `ScorerStepByStep` over `datasets/train.parquet`.  
- Stream through all rows with the `PredictionModel` in `solution.py`.  
- Print mean R² across all 32 features and a few per-feature R²s.  

Use this to sanity-check runtime and streaming behaviour before packaging a submission.

### 2.4 Build MLP Submission ZIP

The competition expects a ZIP with `solution.py` at the root and any required model files alongside it.
We keep these files in a dedicated **submissions folder** at the project root so that each submission version is self-contained.

From the project root:

```bash
cd /Users/sergei/PycharmProjects/WunderSex

# Create submissions folder structure
mkdir -p submissions/mlp_v6

# Copy required files from competition_package
cp competition_package/solution.py submissions/mlp_v6/solution.py
cp competition_package/src/utils.py submissions/mlp_v6/utils.py
cp -r competition_package/models submissions/mlp_v6/

# Create the submission ZIP
cd submissions/mlp_v6
zip -r ../submission_mlp_v6.zip .
cd ../..
```

The final ZIP will be at: `submissions/submission_mlp_v6.zip`

Upload this to the competition platform.

> Note: `solution.py` must be at the ZIP root, not inside a nested folder.

---

## 3. CatBoost-Based Submission (MultiRMSE, v5 features)

The CatBoost submission variant lives in `submissions/solution_catboost.py`. It:

- Builds the **v5** lag-based feature set on the fly from the rolling `n_lags` buffer:  
  - Flattened raw lags,  
  - LastKnown-delta,  
  - Mean/std over the lag window,  
  - Lag-1 autocorrelation,  
  - Persistence fraction,  
  - `step_in_seq/1000`.  
- Uses a single `CatBoostRegressor(loss_function="MultiRMSE")` trained offline in `scripts/train_catboost_experiment.py`.  

This is intended as an alternative submission entry; it does not replace the MLP solution.

### 3.1 Train CatBoost (Offline Lab / Submission Model)

From `competition_package`:

```bash
cd competition_package
python scripts/train_catboost_experiment.py
```

This will:

- Load and sort `datasets/train.parquet`.  
- Split by `seq_ix` into train/val (80%/20%).  
- Build supervised `(X, y)` using the same v5 features that `solution_catboost.py` computes online.  
- Train `CatBoostRegressor(loss_function="MultiRMSE")` with a larger number of iterations and early stopping.  
- Save the trained model to:

```bash
models/catboost_lag_delta_multiRMSE.cbm
```

> Training is relatively heavy (hours for the full configuration). Use this as an occasional “oracle” model or for a serious submission, not for fast iteration.

### 3.2 Evaluate Streaming CatBoost on `train.parquet`

From `competition_package`:

```bash
python submissions/solution_catboost.py
```

This will:

- Load `models/catboost_lag_delta_multiRMSE.cbm`.  
- Stream over `datasets/train.parquet` with `ScorerStepByStep`.  
- Print mean R² and a few per-feature R²s for the CatBoost-based `PredictionModel`.  

Use this to check that the CatBoost model behaves correctly in the streaming environment and that runtime is acceptable.

### 3.3 Build CatBoost Submission ZIP

From the project root:

```bash
cd /Users/sergei/PycharmProjects/WunderSex

# Create submissions folder structure
mkdir -p submissions/catboost_v5

# Copy required files from competition_package
cp competition_package/submissions/solution_catboost.py submissions/catboost_v5/solution.py
cp competition_package/src/utils.py submissions/catboost_v5/utils.py
cp -r competition_package/models submissions/catboost_v5/

# Create the submission ZIP
cd submissions/catboost_v5
zip -r ../submission_catboost_v5.zip .
cd ../..
```

The final ZIP will be at: `submissions/submission_catboost_v5.zip`

Upload this to the competition platform.

---

## 4. Recommended Workflow

Putting it together, a practical workflow for new experiments is:

1. **Feature iteration (MLP lab):**
   - Modify feature logic in `scripts/train_model.py` and mirror it in `solution.py` (or use shared `src/features`).  
   - Run `python scripts/train_model.py` to train a new MLP.  
   - Run `python solution.py` to measure streaming R² and runtime.  
   - Log results in `experiments/EXPERIMENT_LOG.md`.  

2. **Occasional CatBoost oracle:**
   - When a feature set looks promising with the MLP, run `python scripts/train_catboost_experiment.py` to train a CatBoost model on the same features.  
   - Use `submissions/solution_catboost.py` to evaluate it in streaming mode.  

3. **Submissions:**
   - Create MLP submissions in `submissions/mlp_v6/` and build ZIPs as `submissions/submission_mlp_v6.zip` for faster, more frequent submissions.
   - Create CatBoost submissions in `submissions/catboost_v5/` and build ZIPs as `submissions/submission_catboost_v5.zip` for occasional CatBoost-based submissions, given its heavy training cost.
   - All submission files and ZIPs are kept in the centralized `submissions/` folder at the project root, separate from the source code.

This keeps the submission process predictable while allowing you to iterate quickly on features and models. 
