# SOTA Ideas Store

This note tracks high‑ROI modeling ideas, their status, and implementation details tailored to our constraints (streaming inference, CPU-only, <60 min).

## Baseline / Context
- **Stable fallback:** v19 lag-MLP (engineered v6 features, 10 lags). LB 0.3563.
- **Current best:** Level + residual blend (α=0.6) on v19 features. LB 0.3571.
- **Constraints:** Streaming `PredictionModel` (step-by-step), CPU-only at inference, PyTorch/NumPy only.

## Strategic Plan (November 2025)

### 1. The "Lean" Hypothesis (Next Step)
All complex models (Regime Blend, Vector Blend, RNN) plateau around 0.356, slightly below the simple scalar blend (0.357).
*   **Diagnosis:** We likely have **Feature Saturation**. 1185 features contain massive redundancy (95%+ correlations). Models are overfitting to noise/collinearity.
*   **Action (v26):** **Aggressive Feature Selection**.
    *   Select Top 100-200 features (Lasso / Importance).
    *   Train "Lean MLP" on this subset.
    *   Goal: Improve generalization by removing noise.

### 2. Regime-Adaptive Blend (Completed)
- **Concept:** Switch Blend Alpha based on detected regime (High Vol -> Level, Trending -> Residual).
- **Result:** LB **0.3565**.
- **Verdict:** Failed to beat scalar blend. Regimes defined on training data don't map cleanly to test set shifts, or "Normal" regime dominates too much.

### 3. Vector Blend (Completed)
- **Result:** LB **0.3566**.
- **Verdict:** Overfit to Pseudo-LB. 32 parameters is too many.

### 4. Stateful Feature-GRU (Completed)
- **Result:** LB **0.3368**.
- **Verdict:** RNNs overfit training regimes.

### 5. TSMixer (The New Frontier)
We have a strong pseudo-LB (0.3837) but weak LB (0.3374) on v2 (Raw+Delta).
- **Status:** v4 Hybrid (Raw+Delta+Stats) achieves Pseudo-LB **0.3957** (Matches MLP!).
- **Diagnosis:** Likely suffering from **distribution shift** (trained on specific price levels).
- **Action Plan (Tier 1 - Do Next):**
    1.  **Input Stem (1x1 Conv):** Compress 192 hybrid features -> 64/96 dims before mixing. Reduces params, forces feature extraction.
    2.  **Robust RevIN-Lite:** Implement Smoothed RevIN (mix window stats with global stats) to handle distribution shift without destabilizing on short windows.
    3.  **Training Recipe:** Add EMA (Exponential Moving Average) and AdamW to stabilize generalization.

**Tier 2 (If Tier 1 fails):**
- **Ensemble Distillation:** Train a small TSMixer to mimic the large Triplet Ensemble.
- **Frequency Mixing:** Replace Time-MLP with FFT (FNet style) for noise filtering.

---

## Tested / Deprioritized
- **v28 Hybrid Ensemble (MLP + MLP + TSMixer v4)**
  - Status: Ready for Submission (Next Day). Pseudo-LB ~0.40+.
- **v27 Triplet Blend (MLP Level + MLP Resid + TSMixer v2)**
  - Status: Submitted (Pending Score).
- **v26 TSMixer v2 (Raw + Delta)**
  - Status: LB 0.3374. Good baseline, needs RevIN.
- **v25 Regime-Adaptive Blend**
  - Status: LB 0.3565. Complexity did not pay off.
- **v24 Scalar Blend (0.55/0.45)**
  - Status: LB 0.3564. Robust, but slightly worse than 0.6/0.4 blend (0.3571).
- **v23 Stateful Feature-GRU**
  - Status: LB 0.3368. Good tech, bad generalization.
- **v22 Vector Blend**
  - Status: LB 0.3566. Overfit.
- **v21 MLP (Spreads + Residuals)**
  - Status: LB 0.3451. Explicit spreads hurt generalization.
- **Micro-Mamba**
  - Status: LB 0.1215. Arch mismatch.
- **Triplets / Wavelets**
  - Status: LB 0.3549.

## Exit Criteria
- If "Lean MLP" (v26) also fails to beat 0.357, declare 0.3571 as the **Feature Ceiling** for this dataset/constraints. Focus on submitting the best blend.
