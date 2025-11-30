import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

# Add project root folder to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

# --- MLP Architecture ---
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

# Reuse build_features from calibrate_validation.py manually to avoid import circulars
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

def build_supervised_dataset(df, clip_min, clip_max):
    feature_cols = [str(i) for i in range(32)]
    X_list = []
    y_list = []
    
    N_LAGS = 10
    
    for seq_ix, df_seq in df.groupby("seq_ix"):
        df_seq = df_seq.sort_values("step_in_seq")
        states = df_seq[feature_cols].values
        steps = df_seq["step_in_seq"].values
        need_pred = df_seq["need_prediction"].values
        
        T = len(df_seq)
        for idx in range(T - 1):
            if not need_pred[idx]: continue
            if idx < N_LAGS - 1: continue
            
            raw_window = states[idx - N_LAGS + 1 : idx + 1]
            win = np.clip(raw_window, clip_min, clip_max)
            
            lag_flat = win.reshape(-1)
            last = win[-1]
            delta_flat = (win - last).reshape(-1)
            mean_last = win.mean(axis=0)
            std_last = win.std(axis=0)
            
            ac1 = _compute_lag1_autocorr(win)
            ac2 = _compute_lagk_autocorr(win, 2)
            ac3 = _compute_lagk_autocorr(win, 3)
            acf_sum = np.abs(ac1) + np.abs(ac2) + np.abs(ac3)
            frac = _compute_frac_above_mean(win, mean_last)
            q25, med, q75, iqr, skew, kurt, cv = _compute_robust_window_stats(win, mean_last, std_last)
            slope, r2, curve = _compute_trend_features(win)
            step_val = np.array([steps[idx] / 1000.0], dtype=np.float32)
            
            features = np.concatenate([
                lag_flat, delta_flat, mean_last, std_last,
                ac1, ac2, ac3, acf_sum, frac,
                q25, med, q75, iqr, skew, kurt, cv,
                slope, r2, curve, step_val
            ])
            X_list.append(features)
            y_list.append(states[idx+1])
            
    return np.array(X_list), np.array(y_list)

def main():
    print("Loading Hard Split Data...")
    dataset_path = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")
    df = pd.read_parquet(dataset_path)
    
    hard_split_path = os.path.join(CURRENT_DIR, "..", "datasets", "hard_validation_split.json")
    with open(hard_split_path, "r") as f:
        hard_seqs = json.load(f)
        
    df_hard = df[df["seq_ix"].isin(hard_seqs)].copy()
    print(f"Hard sequences: {len(hard_seqs)}, Samples: {len(df_hard)}")
    
    models_dir = os.path.join(CURRENT_DIR, "..", "models")
    norm = np.load(os.path.join(models_dir, "lag_mlp_normalization.npz"))
    x_mean = norm["x_mean"].astype(np.float32)
    x_std = norm["x_std"].astype(np.float32)
    clip_min = norm["clip_min"].astype(np.float32)
    clip_max = norm["clip_max"].astype(np.float32)
    
    print("Building dataset for MLP...")
    X_hard, y_hard = build_supervised_dataset(df_hard, clip_min, clip_max)
    
    X_norm = (X_hard - x_mean) / x_std
    
    models = []
    device = torch.device("cpu")
    for f in sorted([x for x in os.listdir(models_dir) if x.startswith("lag_mlp_fold") and x.endswith(".pth")]):
        ckpt = torch.load(os.path.join(models_dir, f), map_location=device)
        m = LagMLP_v19(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["output_dim"])
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        
    print("Evaluating MLP v19 on Hard Split...")
    X_tens = torch.from_numpy(X_norm).float()
    preds_accum = np.zeros_like(y_hard)
    
    with torch.no_grad():
        for m in models:
            preds_accum += m(X_tens).numpy()
    preds = preds_accum / len(models)
    
    r2s = [r2_score(y_hard[:, i], preds[:, i]) for i in range(32)]
    mean_r2 = np.mean(r2s)
    
    print(f"MLP v19 Mean R2 on Hard Split: {mean_r2:.5f}")

if __name__ == "__main__":
    main()
