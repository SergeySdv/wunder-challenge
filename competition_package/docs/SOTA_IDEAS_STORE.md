# SOTA Ideas Store

This note tracks high‑ROI modeling ideas, their status, and implementation details tailored to our constraints (streaming inference, CPU-only, <60 min).

## Baseline / Context
- **Stable fallback:** v19 lag-MLP (engineered v6 features, 10 lags). LB 0.3563.
- **Current best:** Level + residual blend (α=0.6) on v19 features. LB 0.3571.
- **Constraints:** Streaming `PredictionModel` (step-by-step), CPU-only at inference, PyTorch/NumPy only.

## Tested / Deprioritized
- **Triplet Imbalance + WVTR (v20)**  
  - Status: Tested. CV/Pseudo-LB ≈ flat; LB 0.3549 (< v19). Deprioritized.
- **GRU sequence model (v12)**  
  - Status: Tested on raw sequences. CV ~0.316, Pseudo-LB ~0.350 (worse than MLP). Deprioritized.
- **NLinear long-context (v16)**  
  - Status: Tested. Large regression (CV ~0.267). Deprioritized.
- **CatBoost on v19 features**  
  - Status: Trails MLP; heavy to train; not a submission candidate.

## Active Candidate: Micro-Mamba (SSD-style SSM)
**Goal:** Lightweight SSM (Mamba-2/SSD-inspired) that runs on CPU in streaming mode and can beat the GRU baseline and, if possible, the v19 MLP.

- **Inputs:** Raw 32-dim values, window 30–50. Optional residual target (y − x_t) to stabilize learning; RevIN (instance norm) on inputs/outputs.
- **Block (2–3 layers):**
  - SSM core: scalar/diagonal decay per head (SSD-style), state dim `d_state` 16–32.
  - Width: `d_model` 64 (try 128 if runtime allows); `nheads`=4, `headdim`=16.
  - Local conv: small causal/depthwise conv (k=4–7) + pointwise mix for short-term patterns.
  - Gating: sigmoid gate to mix SSM and conv outputs.
  - Residual + small FFN; RMSNorm or LayerNorm; SiLU activation; dropout ~0.1 if needed.
- **Training:**
  - Loss: Huber (delta=1.0) to tame outliers; targets as residuals (optional).
  - Optim: Adam, lr 1e-3 (back off to 5e-4 if spiky); weight_decay 0.05–0.1; grad clip 1.0.
  - Split: same pseudo-LB split (10% seqs) + val split inside dev (held-out seqs). Early stop on val R² (patience ~5–8).
  - Quick pilot: subset 80k–120k samples to see if pseudo-LB > GRU (~0.35). If promising, full train.
- **Inference:**
  - Pure PyTorch; no Triton/external kernels. `torch.jit.script` the step to reduce Python overhead.
  - Maintain SSM state and tiny conv buffer across steps; float32; batch=1; CPU-only.
  - Optional ensemble: blend residual Mamba output with v19 level MLP (alpha sweep) if it beats baseline.

## Backlog / Other Ideas
- **LSTM small baseline:** Not yet tried; could mirror the LSTM script on raw window 30–50 with RevIN + residual target for comparison.
- **Hybrid residual head:** Train SSM to predict residuals of v19 MLP; blend at inference if it helps Pseudo-LB.
- **SimCLR-Time / contrastive pretrain:** Still backlog; heavier lift, lower priority under current CPU budget.
- **JointSDAE / autoencoder-style features:** Backlog; deprioritized until SSM verdict.

## Exit Criteria
- Keep Micro-Mamba only if Pseudo-LB lifts ≥0.002 over v19 (or clear LB gain). Otherwise, stick with v19/blend.
