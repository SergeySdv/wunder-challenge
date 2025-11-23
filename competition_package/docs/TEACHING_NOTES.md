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

### 7.1 How We Check CatBoost for Overfitting

A common question is: **“Is CatBoost just overfitting, or is it genuinely slightly worse than the MLP here?”**

In this project we use three simple checks:

1. **Train vs validation loss / R² (from `train_catboost_experiment.py`)**
   - CatBoost prints `learn` (train loss) and `test` (validation loss) at each iteration.  
   - Overfitting pattern:
     - Train loss keeps going down,
     - Validation loss goes down for a while, then starts to **go back up** (best iteration is earlier than the final one).  
   - In our v5 CatBoost runs, the best iteration is near the end and `learn` is **still higher** than `test` at that point, which is the opposite of classical overfitting — the model is simply not as strong as the MLP on this feature set.

2. **Explicit train vs validation R²**
   - After training we can compute:
     - `train_mean_r2 = R²(model(X_train), y_train)`  
     - `val_mean_r2 = R²(model(X_val), y_val)`  
   - If `train_mean_r2` ≫ `val_mean_r2`, that’s a sign of overfitting.  
   - In our experiments, CatBoost v5 shows **similar train and val R²**, reinforcing that the gap to the MLP is mainly a **bias / feature** issue, not severe overfit.

3. **Streaming check on held-out sequences**
   - For the submission-style `PredictionModel` (see `solution_catboost.py`), we run the **streaming scorer** on:
     - The full train file (`train.parquet`), and  
     - Optionally, only the validation `seq_ix` subset.  
   - If the streaming R² drops a lot only on the held-out sequences (while being very high on the sequences used for training the trees), that suggests overfitting to the training sequences.

In our current runs:

- CatBoost v5’s val mean R² (~0.4275) is **slightly below** the MLP v5/v6 (~0.431–0.432),  
- Streaming R² on the full train file (~0.365) is roughly in line with the MLP,  
- There is **no strong train≫val gap**.

So the conclusion is: the CatBoost model is **not wildly overfitted**, it is just a bit less well‑matched to our engineered feature space than the Lag‑MLP, while being much more expensive to train in this configuration.

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

---

## 11. Advanced Experiments (v10–v13)

After stabilizing the baseline, we ran a series of "advanced" experiments to push the leaderboard score.

### 11.1 Robustness & Winsorization (v10)
**Goal:** Fix the disconnect between Train R² and Leaderboard scores.
**Idea:**
- **Surge Filtering:** Markets have rare 10σ events. These "spikes" can wreck neural net gradients.
- **Winsorization:** We clipped all inputs to the **[0.1%, 99.9%]** quantiles learned from training data.
- **Strict Validation:** Moved from a single 80/20 split to **5-Fold CV + Pseudo-LB** (holding out 10% of sequences completely).
**Result:** Pseudo-LB score jumped to **0.39**, and Leaderboard score improved to **0.3513**. Robustness matters!

### 11.2 Hyperparameter Tuning (v11)
**Goal:** Squeeze the MLP.
**Idea:** Used **Optuna** to randomly search hidden sizes, dropout rates, and learning rates.
**Finding:** A **smaller** model (128 hidden units vs 256) with **higher dropout** (0.3 vs 0.1) generalized better. This confirms the dataset is noisy and prone to overfitting.
**Result:** Leaderboard score: **0.3540** (Current Best).

### 11.3 Failed "Big Ideas" (v12 & v13)
We tried to beat the "feature engineering" ceiling with smarter architectures:
1.  **v12 GRU (Recurrent Net):** Fed full sequences to a GRU to learn temporal dynamics automatically.
    *   **Result:** Failed (LB ~0.350).
    *   **Lesson:** On small/noisy data, explicit features (rolling stats) beat "black box" temporal learning.
2.  **v13 Kinematics:** Added physics-based features (Acceleration, Path Roughness).
    *   **Result:** Regression (LB ~0.3529).
    *   **Lesson:** Adding 100+ complex features just added noise. Simpler features are more robust.

We reverted to **v11** as the stable "Gold Standard". Simplicity + Robustness wins.

---

# 🎓 Teacher's Notes: The Quest for the Top 10

We already had a good model (a simple Neural Network called an "MLP"). It was doing okay, but we wanted to break into the top leaderboard positions. To do that, we tried four advanced strategies. Two worked, and two failed.

Here is the story of why.

---

## 1. The "Loud Noise" Problem (Experiment v10)
**Technique: Winsorization & Robust Validation**

Imagine you are recording a podcast. Most people speak at a normal volume. But occasionally, someone drops a microphone or screams. That huge spike in volume distorts the whole recording.

Financial data is the same. Most price changes are small ($100 \to $101). But sometimes, a "Flash Crash" happens ($100 \to $50 in one second).

**The Problem:**
When our Neural Network sees that huge crash, it panics. It tries to adjust its weights drastically to fix that one error, which ruins its ability to predict normal days. This is called "exploding gradients."

**The Solution (Winsorization):**
We applied a strict filter. We calculated the **0.1%** lowest and **99.9%** highest values in the training history.
*   If a value is higher than the 99.9% limit, we clamp it down to that limit.
*   We essentially told the model: *"Ignore the crazy extremes. Focus on the normal range."*

**The Validation Upgrade:**
We also stopped trusting a single test. Previously, we split the data once (80% train, 20% test). But what if that 20% was just a really easy (or hard) year?
*   We switched to **5-Fold Cross-Validation**: We split the data into 5 chunks and trained 5 separate models, rotating which chunk was the test set.
*   We averaged their predictions. This is like asking 5 experts for their opinion instead of just one.

**Result:** ✅ **Success!** The model became much more stable and our score jumped.

---

## 2. The "Goldilocks" Principle (Experiment v11)
**Technique: Hyperparameter Tuning**

We had a model with 256 "neurons" (brain cells) in its hidden layer. We assumed "bigger is better," right?

**The Problem (Overfitting):**
A big brain can memorize answers instead of learning rules. If the model memorizes the training data too well, it fails when it sees new data it hasn't seen before.

**The Solution:**
We used a tool called **Optuna** to try hundreds of random combinations of settings.
*   It found that a **smaller brain** (128 neurons) was better.
*   It found that we needed **higher Dropout** (0.3). Dropout is a technique where we randomly turn off 30% of the neurons during training. It forces the remaining neurons to work harder and learn robust patterns, rather than relying on a specific neighbor.

**Result:** ✅ **Success!** This simpler, disciplined model gave us our best score ever (**0.3540**).

---

## 3. The "Smart Student" Trap (Experiment v12)
**Technique: GRU (Recurrent Neural Network)**

Our MLP model only looks at the last 10 steps. It has "amnesia" about anything that happened before that.
We thought: *"Let's use a GRU! A GRU is a memory network that can remember the entire history of 1000 steps."*

**The Hypothesis:**
The GRU should be smarter because it reads the whole history book, not just the last page.

**The Reality (Failure):** ❌
The GRU performed **worse** than the simple MLP. Why?
*   **Noise vs. Signal:** Financial data is incredibly noisy. A model that looks at 1000 steps sees 10 steps of signal and 990 steps of noise. The GRU got confused by the noise.
*   **Feature Engineering:** Our MLP was "fed" specific, hand-crafted summaries (Rolling Mean, Volatility). The GRU had to figure those out by itself from raw numbers. In small datasets, **giving the model the answer (explicit features)** usually beats letting it figure it out (implicit learning).

**Lesson:** Don't use a complex Deep Learning model just because it sounds cool. Sometimes a simple model with good notes works better.

---

## 4. The "Information Overload" (Experiment v13)
**Technique: Kinematics Features**

Since the GRU failed, we went back to the MLP and tried to feed it *more* notes. We calculated Physics-style features:
*   **Acceleration:** Is the price speeding up?
*   **Path Roughness:** Did the price go straight up, or did it zig-zag?

**The Reality (Failure):** ❌
The score barely moved (it actually got slightly worse).

**The Reason:**
We added ~100 new features. Most of them were highly correlated with features we already had (like "Curvature").
*   If you give a student 10 useful facts, they learn well.
*   If you give them 10 useful facts + 100 random trivia facts, they get distracted.
*   We diluted the signal. The model struggled to find the "needle in the haystack" because we added more hay.

---

## 5. The "Last Mile" Optimization (Experiment v19)
**Technique: Bayesian Optimization (Optuna) - Round 2**

We knew v11 was good, but was it *optimal*? We decided to run a much deeper, smarter search.

**The Process:**
*   We used **Tree-structured Parzen Estimator (TPE)**: A smart algorithm that "learns" which hyperparameters work best as it goes.
*   We ran **50 trials** instead of 20.
*   We tuned **Weight Decay** (L2 Regularization) for the first time.

**The Finding:**
The optimizer found a slightly different "sweet spot":
*   **Hidden Size:** 192 (vs 128). Slightly larger capacity.
*   **Dropout:** 0.2 (vs 0.3). Slightly less noise.
*   **Learning Rate:** 1.6e-4 (vs 5e-4). **Much slower learning.**

**The Result:** ✅ **New Personal Best!**
This combination pushed our score to **0.3563**. It turns out that training a slightly larger model *slower* allows it to settle into a better, more generalizable minimum.

---

## Final Summary

We learned that for this specific challenge:
1.  **Clean Data** (Winsorization) is more important than fancy models.
2.  **Simplicity** (Smaller, tuned MLP) beats Complexity (GRU).
3.  **Quality over Quantity** (Selected features > All possible features).
4.  **Patience** (Slower learning rate) pays off in the end.

We are now sticking with **v19** because it represents the perfect balance of robustness and accuracy.

---

## 6. Evaluating "SOTA" Recommendations (The Reality Check)

We consulted a Deep Research report suggesting that simple linear models (**NLinear**) or Gradient Boosting (**LightGBM**) should beat complex MLPs. We tested this rigorosuly:

1.  **NLinear (v16):**
    *   **Theory:** Financial data has long trends; a simple linear map over 336 steps should capture them robustly.
    *   **Reality:** Failed badly (CV ~0.26).
    *   **Lesson:** Our dataset is dominated by **short-term, non-linear microstructure** (lags 1-10). A linear model over 336 steps is too rigid to capture these quick flips.

2.  **Gradient Boosting (CatBoost v17/v18):**
    *   **Theory:** Trees handle noise and outliers better than Neural Nets.
    *   **Reality:** Failed (CV ~0.32).
    *   **Lesson:** While robust, Trees struggled to exploit the shared structure across the 32 targets. The MLP's dense layers learned a better shared representation.

**Conclusion:**
Benchmarks on "Exchange Rate" datasets don't always transfer to specific hackathon data. **Empirical testing > Theory.** Our **Lag-MLP (v19)** remains the champion because it treats the problem as "Tabular Regression with Short Memory," which fits the actual data physics best.
 
