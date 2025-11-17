# catch22 Features – A Student's Guide 📚

## Table of Contents
1. [Introduction to catch22](#introduction-to-catch22)
2. [Why We Use catch22 (Offline Lab Only)](#why-we-use-catch22-offline-lab-only)
3. [Feature Importance in Our Project](#feature-importance-in-our-project)
4. [Detailed Feature Explanations](#detailed-feature-explanations)
   - [Spectral Features](#spectral-features)
   - [Autocorrelation Features](#autocorrelation-features)
   - [Local Prediction Features](#local-prediction-features)
   - [Time-Reversibility Features](#time-reversibility-features)
   - [Persistence Features](#persistence-features)

---

## Introduction to catch22

**catch22** stands for "CAnonical Time-series CHaracteristics" and provides 22 summary statistics that capture different properties of time series data.

### What Problem Does It Solve?

When working with time series, we often need to understand:
- Is the series smooth or noisy?
- Does it have periodic patterns?
- How quickly does it "forget" its past values?
- Are there long persistent trends?
- Is it predictable or chaotic?

Instead of manually engineering hundreds of features, catch22 gives us 22 carefully chosen statistics that capture these properties efficiently.

### The 22 Features Overview

catch22 includes features that measure:
1. **Spectral properties** – frequency content and oscillations
2. **Autocorrelation** – memory and dependence over time
3. **Local predictability** – how well immediate past predicts the future
4. **Distribution properties** – outliers, skewness, entropy
5. **Nonlinear dynamics** – chaos, complexity, time-reversibility
6. **Persistence** – regime behavior and long runs

---

## Why We Use catch22 (Offline Lab Only)

In our Wunder Challenge project:

- **v3 features** (our baseline): lags + LastKnown-delta + rolling mean/std + step → **705 dimensions**
- **v4 features** (lab experiment): v3 + per-sequence catch22 → **1409 dimensions** (705 + 704)

### Key Finding from CatBoost Feature Importance:
- Base v3 features: **~88%** of model importance
- catch22 block: **~12%** of model importance

### Why Not Use in Submission?

⚠️ **Important**: Per-sequence catch22 features use the **entire 1000-step sequence** to compute statistics. This means:
- They encode full-sequence information (future leakage)
- They encode sequence identity from train.parquet
- They **do not generalize** to new hidden test sequences

**Our Strategy**:
1. Use catch22 offline to **discover** which dynamics matter
2. Design **streaming-safe analogs** that work on short lag windows
3. Keep heavy catch22 computation strictly offline

---

## Feature Importance in Our Project

From `catch22_feature_importance.py` analysis:

### Top 10 Most Important catch22 Features:
1. `SP_Summaries_welch_rect_area_5_1` – spectral energy in mid-frequency band
2. `SP_Summaries_welch_rect_centroid` – spectral centroid (frequency center of mass)
3. `CO_f1ecac` – autocorrelation decay time
4. `CO_FirstMin_ac` – first autocorrelation minimum (oscillation scale)
5. `FC_LocalSimple_mean1_tauresrat` – local predictability ratio
6. `FC_LocalSimple_mean3_stderr` – local prediction error
7. `CO_trev_1_num` – time-reversibility statistic
8. `SB_BinaryStats_mean_longstretch1` – longest run above/below mean
9. `CO_HistogramAMI_even_2_5` – auto-mutual information
10. Various distribution and entropy features

Let's dive into each category with detailed explanations.

---

## Detailed Feature Explanations

---

### Spectral Features

Spectral features analyze the **frequency content** of the time series. Think of them as asking: "How much does the series wiggle at different speeds?"

---

#### 1. `SP_Summaries_welch_rect_area_5_1`
**Spectral Power in Mid-Frequency Band**

**What it measures**: Energy (power) concentrated in a specific mid-frequency range of the series.

**Mathematical Background**:
- Uses the **Welch periodogram** method to estimate the power spectral density
- Integrates power over a specific frequency band
- Rectangular window means each data point is weighted equally

**Intuitive Explanation**:

Imagine playing a musical note:
- **Low frequencies** → slow, deep bass notes (long-term trends)
- **Mid frequencies** → melody notes (regular oscillations)
- **High frequencies** → high-pitched, rapid notes (noise, quick changes)

This feature measures how much "mid-frequency melody" is in your time series.

**Example Scenarios**:

```
Series A (smooth trend):
━━━━━━━━━━━━━━━━━━━━
     ╱
    ╱
   ╱
  ╱
 ╱
╱
SP_welch_area_5_1: LOW (mostly low frequencies)

Series B (regular oscillations):
    ╱╲    ╱╲    ╱╲
   ╱  ╲  ╱  ╲  ╱  ╲
  ╱    ╲╱    ╲╱    ╲
━━━━━━━━━━━━━━━━━━━━
SP_welch_area_5_1: HIGH (strong mid-frequency content)

Series C (random noise):
  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲
 ╱  ╲╱  ╲╱  ╲╱  ╲╱
━━━━━━━━━━━━━━━━━━━━
SP_welch_area_5_1: HIGH but spread across many frequencies
```

**How to Interpret**:
- **High value** → Series has significant oscillatory behavior at mid-range periods
- **Low value** → Series is either very smooth (slow-varying) or very noisy (high-frequency)

**Visualization You Could Make**:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Example: smooth vs oscillatory series
t = np.linspace(0, 10, 1000)
smooth = np.sin(0.5 * t)  # Low frequency
oscillatory = np.sin(5 * t)  # Mid frequency

# Compute Welch periodogram
f1, psd1 = signal.welch(smooth)
f2, psd2 = signal.welch(oscillatory)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(t, smooth, label='Smooth')
ax1.plot(t, oscillatory, label='Oscillatory')
ax1.legend()

ax2.plot(f1, psd1, label='Smooth PSD')
ax2.plot(f2, psd2, label='Oscillatory PSD')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Power')
ax2.legend()
```

---

#### 2. `SP_Summaries_welch_rect_centroid`
**Spectral Centroid (Center of Mass of Frequencies)**

**What it measures**: The "average frequency" of the series, weighted by power.

**Mathematical Definition**:
$$
\text{Spectral Centroid} = \frac{\sum_i f_i \cdot P(f_i)}{\sum_i P(f_i)}
$$

where $f_i$ are frequencies and $P(f_i)$ is power at each frequency.

**Intuitive Explanation**:

Think of the frequency spectrum as a physical object with mass distributed along it:
- **Low centroid** → most of the "weight" is at low frequencies (slow changes)
- **High centroid** → most of the "weight" is at high frequencies (rapid changes)

It's like asking: "If I balanced this spectrum on a seesaw, where would the balance point be?"

**Example Scenarios**:

```
Low Spectral Centroid (≈ 0.1 Hz):
Power
  |   ██
  |  ███
  |█████
  |█████▂▂▁▁▁___
  └────────────→ Frequency
      ↑ centroid
(Smooth, slow-varying series)

High Spectral Centroid (≈ 5 Hz):
Power
  |          ██
  |        ████
  |    ▁▂██████
  |▁▁▂▃████████
  └────────────→ Frequency
            ↑ centroid
(Noisy, rapidly-varying series)
```

**How to Interpret**:
- **Low centroid (< 1.0)** → Series is smooth and slowly varying (market trend, temperature)
- **High centroid (> 3.0)** → Series is noisy and rapidly changing (sensor noise, high-frequency trading)

**Real-World Examples**:
- **Audio**: Female voice has higher spectral centroid than male voice (higher pitch)
- **Finance**: Bull market has low centroid (smooth trend up); volatile market has high centroid
- **Weather**: Daily temperature has low centroid; minute-by-minute has higher centroid

---

### Autocorrelation Features

Autocorrelation measures how similar a time series is to a lagged version of itself. These features ask: "How long does the series 'remember' its past?"

---

#### 3. `CO_f1ecac`
**First 1/e Crossing of Autocorrelation**

**What it measures**: How many lags it takes for autocorrelation to decay to $1/e \approx 0.368$.

**Mathematical Background**:

The autocorrelation function at lag $\tau$ is:
$$
\rho(\tau) = \frac{\text{Cov}(X_t, X_{t+\tau})}{\text{Var}(X_t)}
$$

We find the smallest $\tau$ where $\rho(\tau) \leq 1/e$.

**Intuitive Explanation**:

Think of memory:
- **Short-term memory** (low CO_f1ecac) → series "forgets" quickly
- **Long-term memory** (high CO_f1ecac) → series remembers its past for a long time

Like asking: "How many time steps until the series becomes mostly independent of its current value?"

**Example Scenarios**:

```
Series A (random noise):
Value: ╱╲╱╲╱╲╱╲╱╲
Time:  0 1 2 3 4

Autocorrelation:
ρ(0) = 1.00
ρ(1) = 0.05  ← drops below 1/e=0.368 immediately
ρ(2) = -0.02
CO_f1ecac ≈ 1 (very short memory)

Series B (smooth trend):
Value: ╱‾‾‾╲___
Time:  0 1 2 3 4

Autocorrelation:
ρ(0) = 1.00
ρ(1) = 0.98
ρ(2) = 0.95
ρ(3) = 0.89
ρ(4) = 0.80
...
ρ(10) = 0.30  ← drops below 1/e here
CO_f1ecac ≈ 10 (long memory)
```

**Visualization Description**:

Plot 1: Time Series vs Lag
```
Autocorrelation ρ(τ)
1.0 |●
    |  ●●
0.368|────●─────── (1/e threshold)
    |      ●●●
0.0 |__________●●●●
    0  5  10  15  20
        ↑ CO_f1ecac
       Lag τ
```

**How to Interpret**:
- **CO_f1ecac = 1-3** → Random or rapidly changing (white noise, stock returns)
- **CO_f1ecac = 5-15** → Moderate memory (daily weather, slow markets)
- **CO_f1ecac > 20** → Very persistent (annual climate cycles, long-term trends)

**Why It Matters for Forecasting**:

Series with **high CO_f1ecac** are easier to forecast because:
- Past values remain relevant for longer
- Trends persist
- Simple lag models work well

Series with **low CO_f1ecac** are harder:
- Past quickly becomes irrelevant
- Need more sophisticated models
- May be unpredictable

---

#### 4. `CO_FirstMin_ac`
**First Minimum of Autocorrelation**

**What it measures**: The lag at which autocorrelation first reaches a local minimum.

**Why This Matters**:

For **oscillatory series** (like sine waves), autocorrelation becomes negative at roughly half the period:

```
Sine Wave:     ╱‾╲  ╱‾╲  ╱‾╲
               ╱   ╲╱   ╲╱   ╲

Autocorrelation:
τ=0 (same phase): ρ=1.0  (perfect correlation)
τ=T/4:           ρ=0.5  (still positive)
τ=T/2:           ρ≈-1.0  (opposite phase) ← first minimum
τ=T:             ρ=1.0  (back in phase)
```

**Example Scenarios**:

```
Series 1: Pure Sine Wave (period = 20)
Value: ╱╲╱╲╱╲╱╲╱╲
CO_FirstMin_ac ≈ 10 (half the period)

Series 2: Monotonic Trend
Value: ╱‾‾‾‾‾‾‾‾
Autocorrelation never becomes negative
CO_FirstMin_ac = NaN or very large

Series 3: Damped Oscillation
Value: ╱╲╱╲ _ _
       ╱  ╲╱╲
CO_FirstMin_ac ≈ 8 (detects the oscillation)
```

**How to Interpret**:
- **CO_FirstMin_ac = 5-20** → Likely has oscillatory pattern with that period
- **CO_FirstMin_ac = 1-3** → Very noisy or anti-persistent
- **Very large or NaN** → Monotonic trend, no oscillations

**Relationship to Signal Processing**:

CO_FirstMin_ac essentially detects the **dominant period** of oscillations. It's related to:
- Fourier analysis (finding peaks in frequency spectrum)
- Peak detection in raw signal
- Cycle counting

---

#### 5. `CO_HistogramAMI_even_2_5`
**Auto-Mutual Information**

**What it measures**: Non-linear dependence between $X_t$ and $X_{t-\tau}$.

**Key Difference from Autocorrelation**:

| Feature | What It Captures |
|---------|------------------|
| Autocorrelation | **Linear** dependence only |
| Auto-Mutual Information | **Any** dependence (linear + non-linear) |

**Example Where AMI Differs**:

```python
# Linear relationship: y = 2x
t = [0, 1, 2, 3, 4]
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
# Correlation = 1.0 ✓
# AMI = high ✓

# Non-linear relationship: y = x²
t = [0, 1, 2, 3, 4]
x = [-2, -1, 0, 1, 2]
y = [4, 1, 0, 1, 4]
# Correlation ≈ 0.0 ✗ (misses the relationship!)
# AMI = high ✓ (detects it!)
```

**Intuitive Explanation**:

AMI asks: "If I know the value at time $t$, how much does that reduce my uncertainty about the value at time $t+\tau$?"

**How to Interpret**:
- **High AMI** → Strong dependence (linear or non-linear)
- **Low AMI** → Values are independent
- **AMI ≫ |Correlation|²** → Non-linear dynamics present

**When It's Useful**:

Systems with non-linear dynamics:
- Chaotic systems (weather, turbulence)
- Regime-switching markets (calm → volatile)
- Biological signals (heart rate, EEG)

---

### Local Prediction Features

These features measure how well we can predict the next value from recent past values using simple local models.

---

#### 6. `FC_LocalSimple_mean1_tauresrat`
**Local AR(1) Predictability Ratio**

**What it measures**: How predictable the series is from a simple 1-step autoregressive model.

**Mathematical Background**:

Fit a local AR(1) model:
$$
X_t = \phi \cdot X_{t-1} + \epsilon_t
$$

The ratio measures:
$$
\text{tauresrat} = \frac{\tau_{\text{model}}}{\sigma_{\epsilon}}
$$

where $\tau$ is a time scale and $\sigma_\epsilon$ is residual error.

**Intuitive Explanation**:

Think of it as asking: "If I use a dead-simple rule 'tomorrow ≈ today × constant', how well does it work?"

**Example Scenarios**:

```
Series A (random walk):
t:     0    1    2    3    4
Value: 0  +1.2 +0.8 +2.1 +1.9

X[t] ≈ X[t-1] works pretty well
FC_LocalSimple_mean1_tauresrat: HIGH

Series B (pure noise):
t:     0    1    2    3    4
Value: 0  +3.2 -2.1 +1.8 -2.9

X[t] has no relation to X[t-1]
FC_LocalSimple_mean1_tauresrat: LOW
```

**How to Interpret**:
- **High ratio** → Series is smooth and locally predictable (good for simple models)
- **Low ratio** → Series is noisy and unpredictable (need complex models or more features)

**Practical Use**:

This feature helps us decide:
- Should I use a simple lag-1 predictor?
- Or do I need a more complex model with longer lags?

If `tauresrat` is high across most features, our lag-based MLP should work well!

---

#### 7. `FC_LocalSimple_mean3_stderr`
**Local AR(3) Standard Error**

**What it measures**: Typical prediction error when fitting a local AR(3) model.

**Model**:
$$
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \phi_3 X_{t-3} + \epsilon_t
$$

This feature is $\sigma_\epsilon = \sqrt{\frac{1}{n}\sum \epsilon_t^2}$.

**Intuitive Explanation**:

"If I use the last 3 values to predict the next one, how big are my errors on average?"

**How to Interpret**:
- **Low stderr** → Last 3 steps are very informative (smooth, predictable)
- **High stderr** → Last 3 steps don't help much (noisy, complex)

**Relationship to Our MLP**:

Our v3 model uses **last 10 lags** to predict next step. If `FC_LocalSimple_mean3_stderr` is:
- **Low** → Even just 3 lags capture most of the signal; our 10-lag model should do great
- **High** → Need more sophisticated features (deltas, rolling stats, longer lags)

---

### Time-Reversibility Features

---

#### 8. `CO_trev_1_num`
**Time-Reversibility Statistic**

**What it measures**: Whether the series looks different when played forwards vs backwards.

**Mathematical Definition**:

Compare distributions of:
- Forward increments: $X_{t+1} - X_t$
- Backward increments: $X_t - X_{t-1}$

If these have different distributions, the series is **time-irreversible**.

**Intuitive Explanation**:

Record your time series as a video. Play it backwards. Does it look strange?

**Examples of Time-Irreversible Processes**:

```
Stock Prices:
- Crashes: sudden drops (≈ -20% in 1 day)
- Recoveries: gradual rises (≈ +2% per day over months)
→ Asymmetric! Time-irreversible.

Playing it backwards:
- Would show sudden jumps up and slow declines
- Looks unnatural!

CO_trev_1_num: HIGH
```

**Examples of Time-Reversible Processes**:

```
Pure Sine Wave:
╱‾╲_╱‾╲_╱‾╲_

Played backwards:
_╱‾╲_╱‾╲_╱‾╲

Looks identical!
CO_trev_1_num: LOW (≈ 0)
```

**Why It Matters**:

Time-irreversibility indicates:
- **Nonlinear dynamics** (not just linear AR process)
- **Asymmetric responses** (e.g., markets fall faster than they rise)
- **Regime changes** (calm → volatile transitions differ from volatile → calm)

**How to Interpret**:
- **CO_trev_1_num ≈ 0** → Symmetric, reversible (sine wave, Gaussian noise)
- **CO_trev_1_num > 0.5** → Asymmetric dynamics present
- **CO_trev_1_num > 1.0** → Strong asymmetry (crashes, regime switches)

---

### Persistence Features

---

#### 9. `SB_BinaryStats_mean_longstretch1`
**Longest Run Above/Below Mean**

**What it measures**: The length of the longest consecutive stretch where values stay above (or below) the mean.

**Algorithm**:
1. Compute mean: $\mu = \frac{1}{n}\sum X_t$
2. Create binary series: $B_t = 1$ if $X_t > \mu$, else $B_t = 0$
3. Find longest consecutive run of 1s (or 0s)

**Example**:

```
Time Series:
t:     0   1   2   3   4   5   6   7   8   9
Value: 2  -1   3   4   5  -2   1  -3  -1  -2
Mean: μ = 0.8

Binary (above mean):
t:     0   1   2   3   4   5   6   7   8   9
B:     1   0   1   1   1   0   1   0   0   0
       ↑       ├───────┤
      run=1    run=3 ← longest!

SB_BinaryStats_mean_longstretch1 = 3
```

**Intuitive Explanation**:

How long does the series tend to stay on one side of its average?

**How to Interpret**:

| Value | Interpretation | Example |
|-------|----------------|---------|
| 2-5 | Noisy, crosses mean often | White noise |
| 10-30 | Moderate persistence | Daily stock returns with trends |
| 50-100 | Strong persistence | Seasonal temperature (summer stays hot for months) |
| > 100 | Very strong trends/regimes | Multi-year bull market |

**Why It's Useful**:

High `longstretch` means:
- Trends persist (momentum strategies work)
- Regime behavior (bull/bear markets, winter/summer)
- Mean-reversion models may be too aggressive

Low `longstretch` means:
- Series oscillates around mean quickly
- Mean-reversion strategies work well
- Trend-following is risky

**Visualization Example**:

```
High Persistence (longstretch = 50):

Value  ┌────────────────╮
      ─┤                │
       └────────────────┴──────────
       ├─ ~50 steps above mean ──┤

Low Persistence (longstretch = 3):

Value  ╱╲╱╲_╱╲_╱╲_╱╲_╱╲_
      ─┤╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱
       (crosses mean frequently)
```

---

## Summary Table: Quick Reference

| Feature | Category | Measures | Low Value | High Value |
|---------|----------|----------|-----------|------------|
| `SP_welch_area_5_1` | Spectral | Mid-freq energy | Smooth/slow trends | Oscillatory |
| `SP_welch_centroid` | Spectral | Avg frequency | Slow-varying | Noisy/rapid |
| `CO_f1ecac` | Autocorrelation | Memory length | Short memory | Long memory |
| `CO_FirstMin_ac` | Autocorrelation | Oscillation period | No oscillations | Clear cycles |
| `CO_HistogramAMI` | Autocorrelation | Nonlinear dep. | Independent | Dependent |
| `FC_mean1_tauresrat` | Predictability | AR(1) fit quality | Unpredictable | Very predictable |
| `FC_mean3_stderr` | Predictability | AR(3) error | Easy to predict | Hard to predict |
| `CO_trev_1_num` | Dynamics | Time-reversibility | Symmetric | Asymmetric |
| `SB_longstretch1` | Persistence | Longest run | Oscillates | Persistent trends |

---

## Designing Streaming-Safe Analogs

Now that we understand what catch22 features measure, we can design **streaming-safe versions** for our submission model:

### Analog Ideas:

| catch22 Feature | Streaming-Safe Analog | Computation |
|-----------------|----------------------|-------------|
| `SP_welch_area` | Rolling variance, abs deviation | `np.var(lag_window)` |
| `SP_welch_centroid` | Ratio of high-freq to low-freq variance | `var(diff(x)) / var(x)` |
| `CO_f1ecac` | Lag-1 autocorrelation | `np.corrcoef(x[:-1], x[1:])[0,1]` |
| `CO_FirstMin_ac` | Lag at first negative autocorr | Check `corrcoef` at lags 2-10 |
| `FC_mean1_tauresrat` | R² of lag-1 linear regression | Simple `(X[t-1], X[t])` fit |
| `CO_trev` | Skewness of increments | `skew(diff(x))` |
| `SB_longstretch1` | Max run length in last 10 steps | Count consecutive above/below |

**Implementation Strategy**:

1. **Add to `train_model.py`** after the v3 feature block:
   ```python
   # Streaming-safe catch22-inspired features
   lag_1_corr = np.corrcoef(lag_slice[:-1].T, lag_slice[1:].T)[0, 1]
   increment_skew = scipy.stats.skew(np.diff(lag_slice, axis=0))
   # ... etc
   ```

2. **Test impact**: Retrain MLP, compare val R² to v3 baseline

3. **Mirror in `solution.py`** if improvement is consistent

---

## Practice Exercises

### Exercise 1: Understanding Spectral Centroid

Given two series:
```
A = [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0, ...]  (smooth sine)
B = [0, 1, -1, 1, -1, 1, -1, 1, -1, 1, ...]      (fast oscillation)
```

**Question**: Which has higher `SP_welch_centroid`?

<details>
<summary>Answer</summary>

**B** has much higher spectral centroid because it oscillates at a higher frequency (alternates every step vs every 8 steps for A).

</details>

---

### Exercise 2: Autocorrelation Memory

Given:
```
Series 1: [1, 0.9, 0.8, 0.7, 0.6, 0.5, ...]  (slow decay)
Series 2: [1, 0.1, -0.05, 0.02, 0, ...]       (fast decay)
```

**Question**: Which has larger `CO_f1ecac`?

<details>
<summary>Answer</summary>

**Series 1** has much larger CO_f1ecac. It takes ~8-10 lags to drop below 0.368, while Series 2 drops immediately after lag 1.

</details>

---

### Exercise 3: Time Reversibility

Consider a stock price that:
- In 1 day drops -20% (crash)
- Takes 30 days to recover +20% (slow grind up)

**Question**: Would `CO_trev_1_num` be high or low?

<details>
<summary>Answer</summary>

**High**. The downward moves are large and sudden (fat-tailed negative), while upward moves are small and gradual. This asymmetry makes the process strongly time-irreversible.

</details>

---

## Further Reading

### Papers:
1. **catch22 Original Paper**: Lubba et al. (2019) "catch22: CAnonical Time-series CHaracteristics"
2. **hctsa (Full 7000+ Features)**: Fulcher & Jones (2017) "hctsa: A Computational Framework"

### Code Repositories:
- Python: `pip install pycatch22`
- Documentation: https://github.com/chlubba/catch22

### Related Concepts:
- **tsfresh**: Another automated feature extraction library
- **Fourier Analysis**: For spectral features
- **Econometrics**: Autocorrelation functions (ACF, PACF)
- **Chaos Theory**: Lyapunov exponents, entropy

---

## Conclusion

catch22 provides a powerful **offline lab** for understanding time-series dynamics. In our project:

1. ✅ We use it to discover which patterns matter (spectral, autocorrelation, persistence)
2. ✅ We measure feature importance with CatBoost to prioritize what to implement
3. ❌ We do NOT use raw catch22 in submission (leaks sequence identity)
4. ✅ We design streaming-safe analogs inspired by catch22 insights

By combining:
- **Domain knowledge** (lags, deltas, rolling stats)
- **Automated discovery** (catch22 feature importance)
- **Engineering discipline** (leak-free, streaming-safe)

We build a strong, generalizable forecasting model! 🚀
