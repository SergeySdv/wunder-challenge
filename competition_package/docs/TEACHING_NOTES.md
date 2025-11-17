# Wunder Challenge – Teaching Notes / Learning Guide 📚

These notes explain step‑by‑step what we did in this project, why, and how the code and math fit together. They’re written as if for a student learning time‑series modeling and competition workflows.

---

## 1. Problem & Data

**Goal.** At each time step where `need_prediction == 1`, we must predict the **next state vector** (32 features) of a market sequence.

Formally, for each sequence `i` and time step `t`:

- Observed state: a 32‑dimensional vector  
  `s[i, t] ∈ R^32`
- We want to predict the next state (one step ahead) as a function of the past:  
  `ŝ[i, t+1] = f(past states) ∈ R^32`

**Data structure** (see `datasets/DATA_DESCRIPTION.md:1`):

- `seq_ix`: sequence ID (`0..516`), 517 sequences.
- `step_in_seq`: step within sequence (`0..999`), 1000 rows per sequence.
- `need_prediction`: 0/1 flag; predictions are scored for steps `100..998`.
- `0`–`31`: 32 numeric features (the state).

**Scoring & streaming** (see `utils.py:1`):

- The scorer iterates row by row through `train.parquet`.  
- For each row, it:
  - Feeds the **current state** into your `PredictionModel.predict`.
  - Uses the **previous prediction** as the target for the current row.
- This means your model must:
  - Maintain internal state per sequence,
  - Reset when `seq_ix` changes,
  - Return `None` when `need_prediction == 0`,
  - Return a `(32,)` vector when `need_prediction == 1`.

R² for each feature `j` is:

`R2_j = 1 - (Σ_t (y[t,j] - ŷ[t,j])^2) / (Σ_t (y[t,j] - mean_j)^2)`

and the final score is the average of `R2_j` over `j = 0..31`.

---

## 2. First Baseline – Moving Average

We started with a **simple moving average** baseline to understand the interface (see `examples/simple/solution.py:1` and the initial version of `solution.py`):

- Keep a list of all past states in the current sequence.
- Reset this list when `seq_ix` changes.
- When `need_prediction == 1`, output the mean of all past states.

Mathematically, if we’ve seen states \(\mathbf{s}_{i,0}, \ldots, \mathbf{s}_{i,t}\), the baseline prediction is:
\[
\hat{\mathbf{s}}_{i,t+1} = \frac{1}{t+1} \sum_{k=0}^{t} \mathbf{s}_{i,k}.
\]

This baseline is:

- Very simple,
- Correct with respect to the interface,
- A good sanity check before building more complex models.

---

## 3. Tsururu Experiment – Univariate CatBoost

To explore time‑series ideas quickly, we used **Tsururu** offline (see `tsururu_experiment.py:1`).

### 3.1 Mapping the competition data to a time series

We focused on **one feature** (column `"0"`) as the univariate target:

- `id` = `seq_ix` (each sequence is a separate time series),
- `date` = `"2000‑01‑01" + step_in_seq` (synthetic daily timestamp),
- `value` = feature `"0"` (our scalar target).

We built a `TSDataset`:

- `target`: `["value"]`, continuous,
- `date`: `["date"]`, datetime,
- `id`: `["id"]`, categorical.

### 3.2 Pipeline & lags

Using `Pipeline.easy_setup`:

- `target_lags = 10` – last 10 values as features:
  \[
  \mathbf{x}_t = (y_{t-9}, \dots, y_{t}) \in \mathbb{R}^{10},
  \]
  where \(y_t\) is the standardized value of the target at time \(t\).
- `date_lags = 1` – simple date‑based features.
- `target_normalizer = "standard_scaler"` – per‑series z‑score:
  \[
  y^{\text{std}}_t = \frac{y_t - \mu}{\sigma}.
  \]

The **training target** is \(y^{\text{std}}_{t+1}\): predict the next standardized value from the past lags.

### 3.3 Model & strategy

- Model: CatBoostRegressor (wrapped by `tsururu.models.boost.CatBoost`).
- Validation: 3‑fold KFold cross‑validation (`KFoldCrossValidator`).
- Strategy: `RecursiveStrategy(horizon=1, history=50)`:
  - Use up to the last 50 points as context,
  - Always predict 1 step ahead.

Result: mean score ≈ **0.9325** (R²‑like) on feature `0`.  
This tells us:

- “Last ~10 lags + simple normalization + gradient boosting” is a **very strong recipe** for this data.

---

## 4. Building Our Own Supervised Dataset

We cannot use Tsururu in the submission, so we built our own **(X, y)** dataset that matches the streaming problem.

See `train_model.py:1`, function `build_supervised_dataset`.

### 4.1 Features (X)

Let \(\mathbf{s}_{i,t} \in \mathbb{R}^{32}\) be the full 32‑dimensional state for sequence `i` at time `t`.

We fix:

- Number of lags: \(K = 10\).

For each sequence and time `t`:

1. We require:
   - `need_prediction == 1` at time `t`,
   - `t` has at least 10 past observations,
   - A next step `t+1` exists.
2. We build a **lag window**:
   \[
   \text{LagWindow}_{i,t} = \left(
   \mathbf{s}_{i,t-K+1}, \dots, \mathbf{s}_{i,t}
   \right) \in \mathbb{R}^{K \times 32}.
   \]
3. We flatten this window:
   \[
   \mathbf{z}_{i,t} = \text{vec}(\text{LagWindow}_{i,t}) \in \mathbb{R}^{K \cdot 32} = \mathbb{R}^{320}.
   \]
4. We add a simple position feature:
   \[
   u_{i,t} = \frac{\text{step\_in\_seq}_{i,t}}{1000.0} \in \mathbb{R}.
   \]
5. Final feature vector:
   \[
   \mathbf{x}_{i,t} = (\mathbf{z}_{i,t}, u_{i,t}) \in \mathbb{R}^{321}.
   \]

### 4.2 Targets (y)

We define the supervised target as the **next state**:
\[
\mathbf{y}_{i,t} = \mathbf{s}_{i,t+1} \in \mathbb{R}^{32}.
\]

This matches the scoring logic: the prediction we output when we see state at time `t` is compared to the true state at `t+1`.

### 4.3 Train/validation split & leakage

We split **by sequence ID** (see `split_by_seq` in `train_model.py:1`):

- Collect all unique `seq_ix`,
- Shuffle,
- Use 80% for training, 20% for validation.

Important:

- No sequence appears in both train and validation ⇒ no cross‑sequence leakage.
- Within a sequence, each \((\mathbf{x}_{i,t}, \mathbf{y}_{i,t})\) uses only information up to time `t` to predict `t+1` ⇒ no temporal leakage.
- Normalization is computed on **train features only**:
  \[
  \mu_j = \mathbb{E}[X_{train, j}], \quad \sigma_j = \sqrt{\text{Var}(X_{train, j}) + \epsilon},
  \]
  and applied to both train and val.

We also wrote a small `leakage_check.py:1` script to randomly verify:

- `target_step == current_step + 1` for samples,
- The lag window in `X` matches the raw last 10 states,
- The target `y` matches the raw next state.

---

## 5. Training a Lag‑MLP (v1: raw lags) and v2: lags + LastKnown‑delta

We then train a **small neural network** to approximate:
`f: R^D → R^32`, where `D` is the feature dimension (321 in v1, 641 in v2).

### 5.1 Normalization

- Compute `x_mean` and `x_std` over `X_train` (per feature).
- Normalize:
  `x_tilde = (x - mu) / sigma`.

These are saved to `models/lag_mlp_normalization.npz` for reuse at inference.

### 5.2 MLP architecture

Defined in `train_model.py:1` as `LagMLP`:

- Input layer: `D` units (321 in v1, 641 in v2, 705 in v3 with rolling stats, 1409 in v4 with catch22, 769 in v5 with streaming‑safe analog features),
- Hidden layer: 64 units + ReLU activation,
- Output layer: 32 units.

In PyTorch notation:

```python
nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
)
```

### 5.3 Loss and optimization

Loss: Mean Squared Error (MSE) per sample:
\[
\mathcal{L} = \frac{1}{32} \sum_{j=1}^{32} (y_j - \hat{y}_j)^2.
\]

We train with:

- Optimizer: Adam (variant of stochastic gradient descent),
- Learning rate: \(10^{-3}\),
- Batch size: 1024,
- Epochs:
  - v1: 5 epochs.
  - v2 (with LastKnown‑delta features): 10 epochs.
  - v3 (lags + LastKnown‑delta + rolling mean/std): still 10 epochs (we keep training budget fixed).
  - v4 (lags + LastKnown‑delta + rolling mean/std + per-sequence catch22 features): also 10 epochs (we change features, not training budget).

### 5.4 Validation metric (R²) and versions

For validation, we compute R² for each feature dimension and take the mean:

\[
R^2_{\text{mean}} = \frac{1}{32} \sum_{j=1}^{32} R^2_j.
\]

On your run, this reached about **0.416** on the validation split.

With the enriched feature set (v2: raw lags + LastKnown‑delta features, input_dim=641), validation mean R² improved to about **0.42+**.

With a further enriched feature set (v3: raw lags + LastKnown‑delta + rolling mean/std over the 10‑step window, input_dim=705), validation mean R² improved again to about **0.428**.

With automated per-sequence catch22 features appended (v4: v3 features + flattened 32×22 catch22 vector per sequence, input_dim=1409), validation mean R² improved further to about **0.4335** **in offline lab training only**. Later leaderboard experiments showed that using these full‑sequence catch22 descriptors directly in the submission model led to overfitting and a large score drop, so catch22 is now treated as an **offline feature lab only**, and the submission model uses catch22 only to inspire streaming‑safe analogs.

With a streaming‑safe analog extension (v5: v3 features + lag‑1 autocorrelation and persistence fraction over the 10‑step window, input_dim=769), validation mean R² improved modestly again to about **0.4318**, and this configuration generalizes well to the public leaderboard.

We then save:

- `models/lag_mlp.pth`:
  - Network weights (`state_dict`),
  - `input_dim`, `output_dim`, `hidden_dim`, `n_lags`.
- `models/lag_mlp_normalization.npz`:
  - `x_mean`, `x_std`, `n_lags`, `feature_cols`.

---

## 6. Streaming Inference in `solution.py`

Now we must run this model in the **online, row‑by‑row scoring setting**. See `solution.py:1`.

### 6.1 Internal state

`PredictionModel` keeps:

- `current_seq_ix`: ID of the sequence we’re currently in,
- `state_history`: list of the last `n_lags` states (each is a 32‑dim np.ndarray).

On a new `seq_ix`, we reset:

```python
def _reset_sequence(self, seq_ix: int) -> None:
    self.current_seq_ix = seq_ix
    self.state_history = []
```

### 6.2 Per‑step logic

When `predict(data_point)` is called:

1. If `seq_ix` changed → reset sequence state.
2. Append `data_point.state` to `state_history`, keep only last 10.
3. If `need_prediction == False` → return `None`.
4. If `need_prediction == True`:
   - If not enough history (len < 10): return `data_point.state` as a graceful fallback.
   - Else:
     - Build the same feature vector as in training **for the current submission version (v3)**:
       - Flatten last 10 states (raw lags),
       - Compute LastKnown‑delta features: subtract the most recent state from all lags and flatten,
       - Compute rolling statistics over the window: per‑feature mean and std over the last 10 steps,
       - Append `step_in_seq / 1000.0`,
       - Normalize with `x_mean`, `x_std`.
     - Feed it through the MLP to get a `(32,)` prediction.

This ensures:

- The **features at inference time match those at training** for the v3 model that backs the current submission.  
- We respect the streaming protocol.
- All computation is CPU‑friendly and deterministic (we fix PyTorch to 1 thread).

Locally, running `python solution.py` evaluates the v3 model on `train.parquet` and prints:

- v1 (raw lags): mean R² ≈ 0.346,
- v2 (lags + LastKnown‑delta): mean R² ≈ 0.352,
- v3 (lags + LastKnown‑delta + rolling mean/std): slightly higher mean R² on the train file (exact value depends on randomness and was not fully captured in the truncated logs, but runtime remains ~20–25 seconds),
- R² per feature for the first few dimensions.

An offline v4 variant (lags + LastKnown‑delta + rolling mean/std + per‑sequence catch22 features) reached mean R² ≈ **0.378** on the train file, but when mirrored into a submission it overfit the public leaderboard (score dropped to ~0.15).  
As a result, v4 is now treated purely as a **lab configuration**; the production `solution.py` sticks to the v3 feature set and uses catch22 only to inspire streaming‑safe analogs, not as direct per‑sequence descriptors in the submission.

Leaderboard:

- v1 (raw lags) submission achieved ~**0.3266**.
- v2 (lags + LastKnown‑delta) submission achieved ~**0.3293**.
- v3 (lags + LastKnown‑delta + rolling mean/std) improved streaming R² and stayed around the same leaderboard range.  
- v4 (lags + LastKnown‑delta + rolling mean/std + per‑sequence catch22) is treated as a **lab‑only** configuration; when used in submission it badly overfit and scored ~0.15.  
- v5 (lags + LastKnown‑delta + rolling mean/std + lag‑1 autocorr + persistence fraction) is the current submission family and achieved a public leaderboard score of about **0.3390**.

---

## 7. Where Tsururu Fits Now

So far, we’ve used Tsururu only to:

- Validate that **lag‑based CatBoost with normalization** can be extremely strong on this dataset (for feature `0`),
- Learn about:
  - `TSDataset` (mapping raw data to time‑series),
  - `Pipeline` & `LagTransformer` (how lags are generated),
  - Different normalizers (StandardScaler, LastKnownNormalizer, etc.),
  - Strategies like `RecursiveStrategy`, `MIMOStrategy`.

**Next learning steps with Tsururu (offline only):**

- Extend to **multivariate targets** (all 32 features).  
- Try different `target_lags`, `history`, and normalizers.  
- Compare CatBoost vs Tsururu’s DL models (DLinear, PatchTST, etc.) on validation R².  
- Then **copy only the best ideas** (lag patterns, normalization) back into `train_model.py` for the final submission.

We also ran a **direct CatBoost MultiRMSE experiment** on our own lag+delta features (see `train_catboost_experiment.py`):

- Same supervised dataset as the Lag‑MLP v2 (10 lags, LastKnown‑delta, step feature),  
- First, we subsampled 120k training samples and trained a compact CatBoost model (`iterations=80`, `depth=6`, `learning_rate=0.05` with early stopping), achieving validation mean R² ≈ **0.38** (below the ~0.42+ of the MLP v2).  
- Then, we trained a larger CatBoost MultiRMSE model on the **full** 371k supervised samples with `iterations=500` (and early stopping, single thread), which reached validation mean R² ≈ **0.423** — roughly matching or slightly exceeding the Lag‑MLP v2, but at the cost of ~4 hours of CPU training time.  

This shows that:

- Our current feature set is usable for both neural nets and gradient boosting.  
- A sufficiently large CatBoost model on these features can reach similar performance to the MLP v2, but training is much more expensive in our single‑threaded, 32‑output setup.  
- For fast iteration and teaching purposes, the Lag‑MLP remains the primary submission model, while CatBoost serves as a strong offline reference and feature‑engineering lab.

---

## 8. Catch22 Feature Insights (Offline Lab)

We also used **catch22** features as an offline lab to understand what kinds of time‑series patterns matter beyond simple lags and rolling stats. For each sequence and each feature dimension, catch22 computes 22 summary statistics; we then trained CatBoost with both:

- v3 features: lags + LastKnown‑delta + rolling mean/std + step, and
- a 704‑dim per‑sequence catch22 block: 32 dimensions × 22 statistics.

CatBoost feature importance on this v4 feature set shows that:

- The base v3 block accounts for ~88% of total importance.
- The catch22 block accounts for ~12% of total importance.
- The most useful catch22 features are mainly **spectral, autocorrelation, persistence and local trend statistics**.

### Key catch22 Features (Summary)

The most important catch22 statistics that emerged:

- **Spectral features**: `SP_welch_rect_area_5_1` (mid-frequency energy), `SP_welch_rect_centroid` (frequency center of mass)
- **Autocorrelation features**: `CO_f1ecac` (memory decay time), `CO_FirstMin_ac` (oscillation period)
- **Predictability features**: `FC_LocalSimple_mean1_tauresrat` (AR(1) fit quality), `FC_LocalSimple_mean3_stderr` (AR(3) error)
- **Dynamics features**: `CO_trev_1_num` (time-reversibility), `CO_HistogramAMI_even_2_5` (nonlinear dependence)
- **Persistence features**: `SB_BinaryStats_mean_longstretch1` (longest run above/below mean)

**📖 For detailed explanations with examples, visualizations, and intuition, see [`CATCH22_FEATURES_GUIDE.md`](CATCH22_FEATURES_GUIDE.md).**

### Why Not Use in Submission?

⚠️ **Important**: Per-sequence catch22 features use the entire 1000-step sequence to compute statistics, which:
- Encodes full-sequence information (future leakage)
- Encodes sequence identity from train.parquet
- Does not generalize to new hidden test sequences

### Streaming-Safe Analog Strategy

Instead, we use catch22 to **guide the design of streaming-safe analog features** that can be computed from the last 10 steps only:

| catch22 Feature | Streaming-Safe Analog |
|-----------------|----------------------|
| `SP_welch_area` | Rolling variance, absolute deviation |
| `CO_f1ecac` | Lag-1 autocorrelation on window |
| `FC_mean1_tauresrat` | R² of lag-1 regression |
| `CO_trev_1_num` | Skewness of increments |
| `SB_longstretch1` | Max run length in last 10 steps |

These analogs can be computed from the same lag buffer that we already maintain in `solution.py`, so they are safe for submission while still reflecting the kinds of patterns catch22 found useful offline.

---

## 9. Suggested Reading / Concepts

If you want to deepen your understanding, look up:

- **Time‑series basics**:
  - Autoregressive models (AR, ARIMA) – idea of predicting using past lags.
  - Stationarity and why we normalize or difference series.
- **Gradient boosting**:
  - CatBoost documentation (search for “CatBoostRegressor tutorial”).
  - Understanding how tree ensembles approximate functions like \(f(\mathbf{x})\).
- **Neural nets for regression**:
  - MLPs: fully connected layers, ReLU, MSE loss.
  - Training with mini‑batch gradient descent and Adam optimizer.
- **R² metric**:
  - Why it measures “fraction of variance explained”.
  - Difference between high R² on train vs generalization.

For Tsururu specifically, you can search:

- “Tsururu time series library tutorial”  
  and follow their notebooks on TSDataset, transformers, and strategies—many of the code snippets in your notebook are directly adapted from those.

---

## 10. Summary for Students

- We started with a **simple, correct baseline** (moving average) to understand the competition API.
- We used **Tsururu** as a sandbox to quickly see that **lag features + normalization + CatBoost** work extremely well for forecasting one feature.
- We then **engineered our own feature pipeline**:
  - Built supervised (X, y) pairs from the streaming data (10 lags + step feature → next state).
  - Carefully avoided data leakage by splitting by `seq_ix` and using only past information.
- We trained a **small MLP** on these features and integrated it into `solution.py` with a streaming `PredictionModel`.
- This yielded a **real, learning‑based model** that is already competitive on the leaderboard and provides a strong foundation for further experiments (more lags, different models, etc.).

The key lesson: **separate exploration from implementation**. Use tools like Tsururu to explore ideas quickly, but always re‑implement the winning ideas in a simple, transparent way that you fully understand and can control. 
