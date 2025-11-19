# Feature Registry 📚

This document serves as the **Single Source of Truth** for all features used in the `WunderSex` project. It maps feature names to their mathematical definitions, motivations, and the experiment version where they were introduced.

**Current Feature Set Version:** v13 (Kinematics & Volatility)
**Total Input Dimension:** ~1281 (v13) / ~1185 (v11)

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

---

## 5. Deprecated / Failed Features

Features tried in experiments but removed due to poor performance or leakage.

| Feature Name | Experiment | Reason for Removal |
| :--- | :--- | :--- |
| **Per-Sequence catch22** | v4 | **Leakage**. Uses future data (entire 1000-step sequence) to compute stats. Overfit LB (0.15). |
| **Residual Targets** | v8 | Predict `y_{t+1} - y_t`. Improved Training R² but degraded Leaderboard (0.3378). Level targets generalize better. |
| **Pair Spreads** | v9 | `x_i - x_j` for correlated pairs. Improved Training R² slightly but added complexity for no LB gain. |
| **SeqGRU Embedding** | v12 | Raw sequence passed to RNN. Failed to beat explicit features (LB ~0.350 vs MLP 0.354). |

---

*Use this registry to check for redundancy before proposing new features.*
