# Feature Registry 📚

This document serves as the **Single Source of Truth** for all features used in the `WunderSex` project. It maps feature names to their mathematical definitions, motivations, and the experiment version where they were introduced.

**Current Feature Set Version:** v19 (Optuna-Tuned MLP; streaming-safe v6 feature set)
**Total Input Dimension:** ~1185 (v19/v11)

---

## 1. Core State Features (v1-v3)

These are the fundamental building blocks of the input vector.

| Feature Name | Dimension | Description / Formula | Motivation | Introduced |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Lags** | `10 × 32` | `[x_{t-9}, ..., x_t]` (flattened) | Captures absolute position history. | v1 |
| **LastKnown Deltas** | `10 × 32` | `x_{t-k} - x_t` for all lags | Captures velocity/displacement relative to current state. Removes level-dependency. | v2 |
| **Rolling Mean** | `32` | `mean(lag_window)` | Local baseline level. | v3 |
| **Rolling Std** | `32` | `std(lag_window)` | **Absolute Volatility**. Measures recent energy/risk. | v3 |
| **Step Position** | `1` | `step_in_seq / 1000.0` | Global position in the sequence (0.0 to 1.0). Helps model learn regime shifts over time. | v1 |

---

## 2. Streaming-Safe Analogs (v5-v6)

Features designed to mimic `catch22` statistics (spectral/autocorr properties) but computable on a short 10-step window.

| Feature Name | Dimension | Description / Formula | Motivation | Introduced |
| :--- | :--- | :--- | :--- | :--- |
| **Lag-1 Autocorr** | `32` | `corr(x_{t-1}, x_t)` | **Memory**. How strongly does the previous step predict the next? | v5 |
| **Lag-2/3 Autocorr** | `32+32` | `corr(x_{t-k}, x_t)` | **Short-term Cycles**. Detects oscillations/reversions over 2-3 steps. | v6 |
| **ACF Sum** | `32` | `sum(|ACF_1..3|)` | **Total Memory Strength**. Aggregate predictability measure. | v6 |
| **Persistence** | `32` | `mean(x > mean(x))` | **Regime Bias**. Fraction of time spent above average (e.g. 0.8 = strong uptrend/high state). | v5 |
| **Robust Quantiles** | `32×3` | `Q25, Median, Q75` | **Robust Distribution**. Less sensitive to outliers than Mean. | v6 |
| **IQR** | `32` | `Q75 - Q25` | **Robust Volatility**. Outlier-resistant spread measure. | v6 |
| **Skewness** | `32` | `mean(((x-μ)/σ)³)` | **Asymmetry**. Detects crashes vs. bubbles (tail risk). | v6 |
| **Kurtosis** | `32` | `mean(((x-μ)/σ)⁴) - 3` | **Fat Tails**. Detects shock-prone regimes. | v6 |
| **Coef of Variation** | `32` | `std / |mean|` | **Relative Volatility**. Risk per unit of value. | v6 |

---

## 3. Trend & Kinematics (v6, v13)

Features derived from physics analogies (Velocity, Acceleration) and geometric curve fitting.

| Feature Name | Dimension | Description / Formula | Motivation | Introduced |
| :--- | :--- | :--- | :--- | :--- |
| **Trend Slope** | `32` | Slope $m$ of `y = mx + c` (Least Squares) | **Average Velocity**. Smoothed direction of travel. | v6 |
| **Trend R²** | `32` | $R^2$ of linear fit | **Trend Quality**. 1.0 = Perfect line, 0.0 = Random cloud. Distinguishes drifts from trends. | v6 |
| **Curvature** | `32` | `Slope(Late) - Slope(Early)` | **Low-Freq Acceleration**. Is the trend speeding up (convex) or slowing down (concave)? | v6 |
| **Mean Acceleration** | `32` | `mean(diff(diff(x)))` | **High-Freq Acceleration**. Instantaneous force acting on the price. | v13 |
| **Volatility Expansion** | `32` | `std(last_5) / std(last_10)` | **Regime Shift**. >1.0 = Volatility exploding. <1.0 = Consolidating. | v13 |
| **Path Roughness** | `32` | `sum(|diff|) / |total_disp|` | **Efficiency Ratio**. 1.0 = Straight line. High = Choppy/Inefficient path. | v13 |

---

## 4. Feature Engineering Implementation

All features are computed in two identical functions to ensure Train/Serve consistency:
1. **Offline**: `train_model.py -> build_supervised_dataset()`
2. **Online**: `solution.py -> PredictionModel._build_features()`

### Preprocessing (Winsorization)
Before ANY feature calculation, the raw input window `x` is **Winsorized** (Clipped) using global quantiles (0.1% - 99.9%) learned from the training set. This prevents "exploding gradients" from rare data spikes.

### Normalization
After feature concatenation, the entire vector is **Standardized** (`(x - mean) / std`) using global statistics learned from the training set.
The normalization file (`models/lag_mlp_normalization.npz`) stores these stats, clip bounds, and `n_lags`. Regenerate it (re-run `train_model.py`) whenever the feature set changes.

---

## 5. Deprecated / Failed Features

Features tried in experiments but removed due to poor performance or leakage.

| Feature Name | Experiment | Reason for Removal |
| :--- | :--- | :--- |
| **Per-Sequence catch22** | v4 | **Leakage**. Uses future data (entire 1000-step sequence) to compute stats. Overfit LB (0.15). |
| **Residual Targets** | v8 | Predict `y_{t+1} - y_t`. Improved Training R² but degraded Leaderboard (0.3378). Level targets generalize better. |
| **Pair Spreads** | v9 | `x_i - x_j` for correlated pairs. Improved Training R² slightly but added complexity for no LB gain. |
| **SeqGRU Embedding** | v12 | Raw sequence passed to RNN. Failed to beat explicit features (LB ~0.350 vs MLP 0.354). |
| **Triplet Imbalance + WVTR Block** | v20 | Added 152 dims; CV/Pseudo-LB flat, streaming R² regressed, LB 0.3549 (< v19 0.3563). Not worth keeping. |

---

## 6. Hybrid TSMixer Features (v26+)

Introduced in **v26 (TSMixer v4 Hybrid)** to bridge the gap between MLP feature engineering and TSMixer's temporal learning. Instead of flattening everything, we stack engineered features as **extra channels** in the `(Time, Channels)` grid.

**Total Channels:** 192 (Input to TSMixer)

| Feature Block | Dimension | Description / Formula | Motivation |
| :--- | :--- | :--- | :--- |
| **Raw Lags** | `10 × 32` | Original state history. | Base signal. |
| **Deltas** | `10 × 32` | `x[t] - x[t-1]` | Velocity/Momentum. Makes model robust to level shifts. |
| **Rolling Mean** | `10 × 32` | `mean(window)` (Broadcasted) | Explicit local level reference. Helps TSMixer normalize internally. |
| **Rolling Std** | `10 × 32` | `std(window)` (Broadcasted) | Volatility signal. |
| **Trend Slope** | `10 × 32` | Linear regression slope (Broadcasted) | Directional trend strength. |
| **Skewness** | `10 × 32` | `((x-μ)/σ)³` (Broadcasted) | Asymmetry/Crash risk signal. |

*Note: Scalar features (Mean, Std, Slope, Skew) are computed once per window and **tiled** (repeated) across the 10 time steps to preserve the 2D grid structure required by TSMixer.*

---

*Use this registry to check for redundancy before proposing new features.*
