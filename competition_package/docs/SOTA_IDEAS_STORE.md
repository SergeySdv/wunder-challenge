# SOTA Ideas Store

This note tracks high‑ROI modeling ideas, their status, and implementation details tailored to our constraints (streaming inference, CPU-only, <60 min).

## Baseline / Context
- **Stable fallback:** v19 lag-MLP (engineered v6 features, 10 lags). LB 0.3563.
- **Current best:** Level + residual blend (α=0.6) on v19 features. LB 0.3571.
- **Constraints:** Streaming `PredictionModel` (step-by-step), CPU-only at inference, PyTorch/NumPy only.

## Strategic Plan (November 2025)

### 1. Vector Blend Optimization (Completed)
- **Results:**
  - Pseudo-LB Mean R² improved to **0.4043** (vs scalar 0.4038).
  - **Public LB:** **0.3566** (Regressed vs Scalar Blend 0.3571).
- **Diagnosis:** 32-parameter optimization overfit the small Pseudo-LB (10% of data). The global scalar alpha is more robust.
- **Status:** Deprioritized. Stick to scalar blend for MLP-only submissions.

### 2. The Structural Fix: Stateful Feature-GRU (High ROI)
Combine strong engineering (v19 features) with infinite memory (RNN) to capture regime changes, fixing the input quality and timeout issues of previous attempts.
*   **Input:** 1185-dim engineered features (not raw data).
*   **Architecture:** `FeatureGRU` (Linear Encoder -> GRU -> Linear Head).
*   **Training:** Feed full sequences (Batch, Seq_Len, 1185), not random windows.
*   **Inference:** **Stateful O(1)**. Pass hidden state `h` from step $t$ to $t+1$. Do not re-process the window.
    *   *Crucial:* Feed zero-features during warmup (steps 0-99) to initialize RNN state.
*   **Target:** ~0.36-0.37 alone, higher when blended.

### 3. Diagnosing the Validation Gap
Address the large gap between Pseudo-LB (~0.40) and Real LB (~0.35).
*   **Hypothesis:** Random split is biased; holdout shares regimes with training.
*   **Action:** Implement **Stratified Splitting by Volatility**.
    *   Sort sequences by std dev of Feature 0.
    *   Pick every 10th sequence for validation.
    *   Ensures validation covers calm, trending, and chaotic regimes equally.

### 4. Step-Dependent Ensembles (Robustness)
Combat non-stationarity within sequences.
*   **Idea:** Train `Model_Early` (steps 100–600) and `Model_Late` (steps 600–999).
*   **Inference:** Switch model based on `step_in_seq`.

---

## Tested / Deprioritized
- **v21 MLP (Spreads + Residuals)**
  - Status: LB 0.3451. Explicit spreads hurt generalization vs v19.
- **Triplet Imbalance + WVTR (v20)**  
  - Status: Tested. CV/Pseudo-LB ≈ flat; LB 0.3549 (< v19). Deprioritized.
- **GRU sequence model (v12)**  
  - Status: Tested on raw sequences. CV ~0.316, Pseudo-LB ~0.350 (worse than MLP). Deprioritized.
- **Micro-Mamba (SSD-style SSM)**  
  - Status: Tested pilots and v19-feature variant (residual targets).  
    - Raw pilots: Val R² 0.326 / 0.264, Pseudo-LB 0.350 / 0.287.  
    - v19 features (window 10, residual, 20k subset): Val R² 0.356, Pseudo-LB 0.383; LB 0.1215.  
  - Underperforms v19 and GRU; deprioritized unless a new design appears.
- **NLinear long-context (v16)**  
  - Status: Tested. Large regression (CV ~0.267). Deprioritized.
- **CatBoost on v19 features**  
  - Status: Trails MLP; heavy to train; not a submission candidate.

## Backlog / Other Ideas
- **LSTM small baseline:** Not yet tried; could mirror the LSTM script on raw window 30–50 with RevIN + residual target for comparison.
- **Hybrid residual head:** Train SSM to predict residuals of v19 MLP; blend at inference if it helps Pseudo-LB.
- **SimCLR-Time / contrastive pretrain:** Still backlog; heavier lift, lower priority under current CPU budget.
- **JointSDAE / autoencoder-style features:** Backlog; deprioritized until SSM verdict.

## Exit Criteria
- Keep Micro-Mamba only if Pseudo-LB lifts ≥0.002 over v19 (or clear LB gain). Otherwise, stick with v19/blend.
