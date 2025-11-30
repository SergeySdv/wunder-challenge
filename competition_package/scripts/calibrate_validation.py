import os
import sys
import numpy as np
import pandas as pd
import torch
import json
from sklearn.metrics import r2_score

# Add project root folder to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from src.models.torchtsmixer import TSMixer # Using TSMixer API but loading MLP logic manually for simplicity?
# Actually, let's just load the MLP v19 manually as we did in solution.py
# to be exact.

class LagMLP_v19(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x): return self.net(x)

# Helpers
def _compute_lag1_autocorr(lag_slice):
    x = lag_slice[:-1, :]
    y = lag_slice[1:, :]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    num = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    denom = np.sqrt(((x - x_mean)**2).mean(axis=0) * ((y - y_mean)**2).mean(axis=0)) + 1e-8
    return num / denom

def _compute_lagk_autocorr(lag_slice, lag):
    if lag <= 0 or lag >= lag_slice.shape[0]: return np.zeros(lag_slice.shape[1], dtype=np.float32)
    x = lag_slice[:-lag, :]
    y = lag_slice[lag:, :]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    num = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    denom = np.sqrt(((x - x_mean)**2).mean(axis=0) * ((y - y_mean)**2).mean(axis=0)) + 1e-8
    return num / denom

def _compute_frac_above_mean(lag_slice, mean_last):
    return (lag_slice > mean_last[None, :]).mean(axis=0)

def _compute_robust_window_stats(lag_slice, mean_last, std_last):
    q25 = np.percentile(lag_slice, 25, axis=0)
    median = np.percentile(lag_slice, 50, axis=0)
    q75 = np.percentile(lag_slice, 75, axis=0)
    iqr = q75 - q25
    denom = std_last + 1e-8
    standardized = (lag_slice - mean_last[None, :]) / denom[None, :]
    skewness = (standardized**3).mean(axis=0)
    kurtosis = ((standardized**4).mean(axis=0) - 3.0)
    cv = std_last / (np.abs(mean_last) + 1e-8)
    return q25, median, q75, iqr, skewness, kurtosis, cv

def _compute_trend_features(lag_slice):
    n_lags, dim = lag_slice.shape
    t = np.arange(n_lags, dtype=np.float32)
    sum_t = float(n_lags * (n_lags - 1) / 2.0)
    sum_t2 = float(n_lags * (n_lags - 1) * (2 * n_lags - 1) / 6.0)
    sum_y = lag_slice.sum(axis=0)
    sum_ty = (t[:, None] * lag_slice).sum(axis=0)
    denom = n_lags * sum_t2 - sum_t * sum_t
    if denom == 0.0:
        slope = np.zeros(dim, dtype=np.float32)
        r2 = np.zeros(dim, dtype=np.float32)
    else:
        slope = (n_lags * sum_ty - sum_t * sum_y) / denom
        intercept = (sum_y - slope * sum_t) / float(n_lags)
        fitted = intercept[None, :] + slope[None, :] * t[:, None]
        residual = lag_slice - fitted
        ss_res = (residual**2).sum(axis=0)
        mean_y = lag_slice.mean(axis=0)
        ss_tot = ((lag_slice - mean_y[None, :]) ** 2).sum(axis=0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    mid = n_lags // 2
    slope_first = (lag_slice[mid, :] - lag_slice[0, :]) / float(mid)
    slope_second = (lag_slice[-1, :] - lag_slice[mid, :]) / float(n_lags - mid)
    curvature = slope_second - slope_first
    return slope, r2, curvature

def build_features(raw_window, step_in_seq):
    lag_flat = raw_window.reshape(-1)
    last = raw_window[-1]
    delta_flat = (raw_window - last).reshape(-1)
    mean_last = raw_window.mean(axis=0)
    std_last = raw_window.std(axis=0)
    
    ac1 = _compute_lag1_autocorr(raw_window)
    ac2 = _compute_lagk_autocorr(raw_window, 2)
    ac3 = _compute_lagk_autocorr(raw_window, 3)
    acf_sum = np.abs(ac1) + np.abs(ac2) + np.abs(ac3)
    frac = _compute_frac_above_mean(raw_window, mean_last)
    q25, med, q75, iqr, skew, kurt, cv = _compute_robust_window_stats(raw_window, mean_last, std_last)
    slope, r2, curve = _compute_trend_features(raw_window)
    
    step_val = np.array([step_in_seq / 1000.0], dtype=np.float32)
    
    return np.concatenate([
        lag_flat, delta_flat, mean_last, std_last,
        ac1, ac2, ac3, acf_sum, frac,
        q25, med, q75, iqr, skew, kurt, cv,
        slope, r2, curve, step_val
    ])

def main():
    print("Loading Data...")
    dataset_path = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")
    df = pd.read_parquet(dataset_path).sort_values(["seq_ix", "step_in_seq"])
    
    print("Loading MLP v19 Models...")
    device = torch.device("cpu")
    models_dir = os.path.join(CURRENT_DIR, "..", "models")
    
    norm = np.load(os.path.join(models_dir, "lag_mlp_normalization.npz"))
    x_mean = norm["x_mean"].astype(np.float32)
    x_std = norm["x_std"].astype(np.float32)
    clip_min = norm["clip_min"].astype(np.float32)
    clip_max = norm["clip_max"].astype(np.float32)
    
    models = []
    for f in sorted([x for x in os.listdir(models_dir) if x.startswith("lag_mlp_fold") and x.endswith(".pth")]):
        ckpt = torch.load(os.path.join(models_dir, f), map_location=device)
        m = LagMLP_v19(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["output_dim"])
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        
    print(f"Loaded {len(models)} models.")
    
    # Score per sequence
    seq_scores = []
    N_LAGS = 10
    feature_cols = [str(i) for i in range(32)]
    
    print("Scoring all 517 sequences...")
    
    for seq_ix, df_seq in df.groupby("seq_ix"):
        # Prepare Batch for this sequence
        df_seq = df_seq.sort_values("step_in_seq")
        states = df_seq[feature_cols].values.astype(np.float32)
        steps = df_seq["step_in_seq"].values
        need_pred = df_seq["need_prediction"].values
        
        # Gather all valid samples
        X_seq = []
        y_seq = []
        
        T = len(df_seq)
        for idx in range(T-1):
            if not need_pred[idx]: continue
            if idx < N_LAGS - 1: continue
            
            raw_window = states[idx - N_LAGS + 1 : idx + 1]
            win = np.clip(raw_window, clip_min, clip_max)
            
            feat = build_features(win, steps[idx])
            X_seq.append(feat)
            y_seq.append(states[idx+1])
            
        if len(X_seq) == 0:
            continue
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Normalize
        X_seq_norm = (X_seq - x_mean) / x_std
        
        # Predict
        t_x = torch.from_numpy(X_seq_norm).float()
        preds = np.zeros_like(y_seq)
        
        with torch.no_grad():
            for m in models:
                preds += m(t_x).numpy()
        preds /= len(models)
        
        # Calc R2 for this sequence
        # Note: R2 can be negative if model is worse than mean.
        # We want "Hardest" = Lowest R2.
        # But careful: single sequence R2 is noisy if variance is low.
        # Let's use MSE as "Hardness"? No, high volatility seqs have high MSE naturally.
        # R2 is better because it's relative to variance.
        
        r2_vals = [r2_score(y_seq[:, i], preds[:, i]) for i in range(32)]
        mean_r2 = np.mean(r2_vals)
        
        seq_scores.append({
            "seq_ix": int(seq_ix),
            "r2": float(mean_r2),
            "mse": float(np.mean((y_seq - preds)**2))
        })
        
    # Convert to DF
    df_scores = pd.DataFrame(seq_scores)
    df_scores = df_scores.sort_values("r2")
    
    print("\n--- Calibration Results ---")
    print(f"Global Mean R2 (Seq-wise average): {df_scores['r2'].mean():.4f}")
    print(f"Worst Sequence R2: {df_scores.iloc[0]['r2']:.4f} (Seq {int(df_scores.iloc[0]['seq_ix'])})")
    print(f"Best Sequence R2: {df_scores.iloc[-1]['r2']:.4f} (Seq {int(df_scores.iloc[-1]['seq_ix'])})")
    
    # Identify Hardest 10%
    n_hard = int(len(df_scores) * 0.10)
    hard_seqs = df_scores.head(n_hard)["seq_ix"].values.tolist()
    
    print(f"\nIdentified {len(hard_seqs)} 'Hard' sequences (Bottom 10% R2).")
    print(f"Mean R2 on Hard Split: {df_scores.head(n_hard)['r2'].mean():.4f}")
    
    # Save
    out_path = os.path.join(CURRENT_DIR, "..", "datasets", "hard_validation_split.json")
    with open(out_path, "w") as f:
        json.dump(hard_seqs, f)
    print(f"Saved hard split to {out_path}")
    
    # Also save full report
    df_scores.to_csv(os.path.join(CURRENT_DIR, "..", "experiments", "sequence_scores_mlp_v19.csv"), index=False)

if __name__ == "__main__":
    main()
