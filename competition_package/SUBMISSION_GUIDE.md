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
python train_model.py
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
python feature_consistency_check.py
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

### 2.4 Build MLP Submission ZIP (using a submission folder)

The competition expects a ZIP with `solution.py` at the root and any required model files alongside it.  
We keep these files in a dedicated **submission folder** under `competition_package` so that each submission version is self-contained.

From the project root:

```bash
cd /Users/sergei/PycharmProjects/WunderSex
cd competition_package

mkdir -p submission_mlp_v6
cp solution.py submission_mlp_v6/solution.py
cp utils.py submission_mlp_v6/
cp -r models submission_mlp_v6/

cd submission_mlp_v6
zip -r ../../submission_mlp_v6.zip .
```

Then go back to the project root (if needed) and upload `submission_mlp_v6.zip` to the competition.  

> Note: `solution.py` must be at the ZIP root, not inside a nested folder.

---

## 3. CatBoost-Based Submission (MultiRMSE, v5 features)

The CatBoost submission variant lives in `competition_package/solution_catboost.py`. It:

- Builds the **v5** lag-based feature set on the fly from the rolling `n_lags` buffer:  
  - Flattened raw lags,  
  - LastKnown-delta,  
  - Mean/std over the lag window,  
  - Lag-1 autocorrelation,  
  - Persistence fraction,  
  - `step_in_seq/1000`.  
- Uses a single `CatBoostRegressor(loss_function="MultiRMSE")` trained offline in `train_catboost_experiment.py`.  

This is intended as an alternative submission entry; it does not replace the MLP solution.

### 3.1 Train CatBoost (Offline Lab / Submission Model)

From `competition_package`:

```bash
cd competition_package
python train_catboost_experiment.py
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
python solution_catboost.py
```

This will:

- Load `models/catboost_lag_delta_multiRMSE.cbm`.  
- Stream over `datasets/train.parquet` with `ScorerStepByStep`.  
- Print mean R² and a few per-feature R²s for the CatBoost-based `PredictionModel`.  

Use this to check that the CatBoost model behaves correctly in the streaming environment and that runtime is acceptable.

### 3.3 Build CatBoost Submission ZIP (using a submission folder)

From the project root:

```bash
cd /Users/sergei/PycharmProjects/WunderSex
cd competition_package

mkdir -p submission_catboost_v5
cp solution_catboost.py submission_catboost_v5/solution.py
cp utils.py submission_catboost_v5/
cp -r models submission_catboost_v5/

cd submission_catboost_v5
zip -r ../../submission_catboost_v5_1.zip .
```

Then upload `submission_catboost_v5_1.zip` (or whatever name you choose) as a CatBoost-based submission.

---

## 4. Recommended Workflow

Putting it together, a practical workflow for new experiments is:

1. **Feature iteration (MLP lab):**
   - Modify feature logic in `train_model.py` and mirror it in `solution.py`.  
   - Run `python train_model.py` to train a new MLP.  
   - Run `python solution.py` to measure streaming R² and runtime.  
   - Log results in `EXPERIMENT_LOG.md`.  

2. **Occasional CatBoost oracle:**
   - When a feature set looks promising with the MLP, run `python train_catboost_experiment.py` to train a CatBoost model on the same features.  
   - Use `solution_catboost.py` to evaluate it in streaming mode.  

3. **Submissions (using submission folders):**
   - Use the MLP folder (e.g. `competition_package/submission_mlp_v6`) to build ZIPs like `submission_mlp_v6.zip` for faster, more frequent submissions.  
   - Use the CatBoost folder (e.g. `competition_package/submission_catboost_v5`) to build ZIPs like `submission_catboost_v5_1.zip` for occasional CatBoost-based submissions, given its heavy training cost.  

This keeps the submission process predictable while allowing you to iterate quickly on features and models. 
