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
We successfully implemented and tuned a TSMixer (MLP-Mixer for Time Series).
- **Status:** v2 (Raw+Delta) achieves Pseudo-LB **0.3837** (strong) but LB **0.3374** (weak).
- **Diagnosis:** Likely suffering from **distribution shift** (trained on specific price levels) and lack of explicit domain features.
- **Next Steps (v28+):**
    1.  **RevIN (Reversible Instance Normalization):** Normalize each input window to mean 0/std 1. Makes the model invariant to absolute price levels. **Crucial** for this dataset.
    2.  **Hybrid TSMixer:** Feed engineered features (Rolling Mean, Volatility) as extra channels. Combines MLP's feature power with TSMixer's grid structure.
    3.  **Patching:** Aggregate time steps into patches (e.g., size 2) to learn local smoothness and reduce noise.

---

## Tested / Deprioritized
- **v27 Triplet Blend (MLP Level + MLP Resid + TSMixer)**
  - Status: Submitted (Pending Score). Ensembles diverse architectures.
- **v26 TSMixer v2 (Raw + Delta)**
  - Status: LB 0.3374. Good baseline, needs RevIN to generalize.
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