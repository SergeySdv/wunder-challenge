import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from typing import Tuple, List, Set, Optional, Dict
import argparse

# --- Configuration ---
N_LAGS = 10
HIDDEN_SIZE = 192  # Optimized by Optuna (was 128)
N_EPOCHS = 25      # Increased for lower LR
BATCH_SIZE = 512   # Optimized
LR = 1.6e-4        # Optimized (was 5e-4)
WEIGHTS_DIR = "models"
PSEUDO_LB_SEED = 999 
CV_FOLDS = 5
CV_SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Helper Functions ---

def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    return df

def compute_winsorization_bounds(df: pd.DataFrame, lower_q=0.001, upper_q=0.999) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = [str(i) for i in range(32)]
    data = df[feature_cols].values
    lower = np.quantile(data, lower_q, axis=0).astype(np.float32)
    upper = np.quantile(data, upper_q, axis=0).astype(np.float32)
    return lower, upper

# --- Feature Engineering ---

def _compute_lag1_autocorr(lag_slice: np.ndarray) -> np.ndarray:
    x = lag_slice[:-1, :]
    y = lag_slice[1:, :]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_center = x - x_mean
    y_center = y - y_mean
    num = (x_center * y_center).mean(axis=0)
    denom = np.sqrt((x_center**2).mean(axis=0) * (y_center**2).mean(axis=0)) + 1e-8
    return (num / denom).astype(np.float32)

def _compute_lagk_autocorr(lag_slice: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0 or lag >= lag_slice.shape[0]:
        return np.zeros(lag_slice.shape[1], dtype=np.float32)
    x = lag_slice[:-lag, :]
    y = lag_slice[lag:, :]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_center = x - x_mean
    y_center = y - y_mean
    num = (x_center * y_center).mean(axis=0)
    denom = np.sqrt((x_center**2).mean(axis=0) * (y_center**2).mean(axis=0)) + 1e-8
    return (num / denom).astype(np.float32)

def _compute_frac_above_mean(lag_slice: np.ndarray, mean_last: np.ndarray) -> np.ndarray:
    above = lag_slice > mean_last[None, :]
    return above.mean(axis=0).astype(np.float32)

def _compute_robust_window_stats(lag_slice: np.ndarray, mean_last: np.ndarray, std_last: np.ndarray):
    q25 = np.percentile(lag_slice, 25, axis=0).astype(np.float32)
    median = np.percentile(lag_slice, 50, axis=0).astype(np.float32)
    q75 = np.percentile(lag_slice, 75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)
    
    denom = std_last + 1e-8
    standardized = (lag_slice - mean_last[None, :]) / denom[None, :]
    skewness = (standardized**3).mean(axis=0).astype(np.float32)
    kurtosis = ((standardized**4).mean(axis=0) - 3.0).astype(np.float32)
    cv = (std_last / (np.abs(mean_last) + 1e-8)).astype(np.float32)
    
    return q25, median, q75, iqr, skewness, kurtosis, cv

def _compute_trend_features(lag_slice: np.ndarray):
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
    
    slope = slope.astype(np.float32)
    r2 = r2.astype(np.float32)

    mid = n_lags // 2
    if mid == 0 or mid == n_lags:
        curvature = np.zeros(dim, dtype=np.float32)
    else:
        first_span = float(mid)
        second_span = float(n_lags - mid)
        slope_first = (lag_slice[mid, :] - lag_slice[0, :]) / first_span
        slope_second = (lag_slice[-1, :] - lag_slice[mid, :]) / second_span
        curvature = (slope_second - slope_first).astype(np.float32)
        
    return slope, r2, curvature

def build_supervised_dataset(
    df: pd.DataFrame, 
    clip_min: np.ndarray, 
    clip_max: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = [str(i) for i in range(32)]
    X_list = []
    y_list = []
    prev_list = []
    
    for seq_ix, df_seq in df.groupby("seq_ix"):
        df_seq = df_seq.sort_values("step_in_seq")
        states = df_seq[feature_cols].values
        steps = df_seq["step_in_seq"].values
        need_pred = df_seq["need_prediction"].values
        
        T = len(df_seq)
        for idx in range(T - 1):
            if not need_pred[idx]:
                continue
            if idx < N_LAGS - 1:
                continue
            
            raw_slice = states[idx - N_LAGS + 1 : idx + 1]
            lag_slice = np.clip(raw_slice, clip_min, clip_max).astype(np.float32)
            
            lag_flat = lag_slice.reshape(-1)
            last = lag_slice[-1]
            delta_slice = lag_slice - last
            delta_flat = delta_slice.reshape(-1)
            
            mean_last = lag_slice.mean(axis=0).astype(np.float32)
            std_last = lag_slice.std(axis=0).astype(np.float32)
            
            ac_lag1 = _compute_lag1_autocorr(lag_slice)
            ac_lag2 = _compute_lagk_autocorr(lag_slice, lag=2)
            ac_lag3 = _compute_lagk_autocorr(lag_slice, lag=3)
            acf_sum_1_3 = (np.abs(ac_lag1) + np.abs(ac_lag2) + np.abs(ac_lag3)).astype(np.float32)
            
            frac_above = _compute_frac_above_mean(lag_slice, mean_last)
            
            q25, median, q75, iqr, skewness, kurtosis, cv = _compute_robust_window_stats(lag_slice, mean_last, std_last)
            trend_slope, trend_r2, curvature = _compute_trend_features(lag_slice)
            step_val = np.array([steps[idx] / 1000.0], dtype=np.float32)
            
            features = np.concatenate([
                lag_flat, delta_flat, mean_last, std_last,
                ac_lag1, ac_lag2, ac_lag3, acf_sum_1_3, frac_above,
                q25, median, q75, iqr, skewness, kurtosis, cv,
                trend_slope, trend_r2, curvature,
                step_val
            ])
            X_list.append(features)
            y_list.append(states[idx+1].astype(np.float32))
            prev_list.append(states[idx].astype(np.float32))
            
    return np.vstack(X_list), np.vstack(y_list), np.vstack(prev_list)

# --- Model Architecture ---

class LagMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Optuna: 0.21
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

# --- Training Engine ---

def train_one_fold(X_train, y_train, X_val, y_val, input_dim, output_dim, fold_idx):
    device = torch.device("cpu")
    model = LagMLP(input_dim, HIDDEN_SIZE, output_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    loss_fn = nn.MSELoss()
    
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    best_val_r2 = -1e9
    best_state = None
    
    print(f"  [Fold {fold_idx}] Training for {N_EPOCHS} epochs...")
    
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
                targets.append(yb.numpy())
        
        preds = np.vstack(preds)
        targets = np.vstack(targets)
        
        r2s = [r2_score(targets[:, i], preds[:, i]) for i in range(targets.shape[1])]
        mean_r2 = np.mean(r2s)
        scheduler.step(mean_r2)
        
        if mean_r2 > best_val_r2:
            best_val_r2 = mean_r2
            best_state = model.state_dict()
            
    print(f"  [Fold {fold_idx}] Best Val R2: {best_val_r2:.5f}")
    return best_state, best_val_r2

# --- Main Pipeline ---

def main():
    parser = argparse.ArgumentParser(description="Train Lag-MLP (v19 features).")
    parser.add_argument("--target_mode", choices=["level", "residual"], default="level", help="Prediction target: next level or next minus current (residual).")
    parser.add_argument("--prefix", default="lag_mlp", help="Prefix for saved model/normalization files.")
    args = parser.parse_args()

    print(f"--- v19 Ultra-Tuned MLP (Optuna Optimized) | target={args.target_mode} | prefix={args.prefix} ---")
    
    dataset_path = os.path.join(BASE_DIR, "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    all_seqs = df["seq_ix"].unique()
    
    rng = np.random.default_rng(PSEUDO_LB_SEED) 
    rng.shuffle(all_seqs)
    
    n_pseudo = int(len(all_seqs) * 0.10)
    pseudo_lb_ids = all_seqs[:n_pseudo]
    dev_ids = all_seqs[n_pseudo:] 
    
    df_pseudo = df[df["seq_ix"].isin(pseudo_lb_ids)].copy()
    df_dev = df[df["seq_ix"].isin(dev_ids)].copy()
    
    print("Computing Winsorization bounds...")
    clip_min, clip_max = compute_winsorization_bounds(df_dev, 0.001, 0.999)
    
    print("Building datasets...")
    X_dev, y_dev_next, prev_dev = build_supervised_dataset(df_dev, clip_min, clip_max)
    X_pseudo, y_pseudo_next, prev_pseudo = build_supervised_dataset(df_pseudo, clip_min, clip_max)

    if args.target_mode == "residual":
        y_dev = y_dev_next - prev_dev
        y_pseudo = y_pseudo_next - prev_pseudo
    else:
        y_dev = y_dev_next
        y_pseudo = y_pseudo_next
    
    x_mean = X_dev.mean(axis=0)
    x_std = X_dev.std(axis=0) + 1e-8
    
    os.makedirs(os.path.join(BASE_DIR, WEIGHTS_DIR), exist_ok=True)
    norm_filename = f"{args.prefix}_normalization.npz" if args.prefix != "lag_mlp" else "lag_mlp_normalization.npz"
    norm_path = os.path.join(BASE_DIR, WEIGHTS_DIR, norm_filename)
    np.savez(
        norm_path, 
        x_mean=x_mean, x_std=x_std, 
        clip_min=clip_min, clip_max=clip_max,
        n_lags=N_LAGS
    )
    
    X_dev_norm = (X_dev - x_mean) / x_std
    X_pseudo_norm = (X_pseudo - x_mean) / x_std
    
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    
    dev_seq_order = sorted(df_dev["seq_ix"].unique())
    seq_sample_counts = []
    for seq_ix, grp in df_dev.groupby("seq_ix"):
        grp = grp.sort_values("step_in_seq")
        need = grp["need_prediction"].values
        count = 0
        T = len(grp)
        for idx in range(T-1):
            if need[idx] and idx >= N_LAGS - 1:
                count += 1
        seq_sample_counts.append(count)
        
    row_seq_ixs = []
    for s_ix, c in zip(dev_seq_order, seq_sample_counts):
        row_seq_ixs.extend([s_ix] * c)
    row_seq_ixs = np.array(row_seq_ixs)
    
    fold_results = []
    input_dim = X_dev.shape[1]
    output_dim = y_dev.shape[1]
    
    print("Starting 5-Fold CV...")
    for fold_i, (train_idx_seq, val_idx_seq) in enumerate(kf.split(dev_ids)):
        train_seqs = dev_ids[train_idx_seq]
        val_seqs = dev_ids[val_idx_seq]
        
        train_mask = np.isin(row_seq_ixs, train_seqs)
        val_mask = np.isin(row_seq_ixs, val_seqs)
        
        X_tr_fold = X_dev_norm[train_mask]
        y_tr_fold = y_dev[train_mask]
        X_val_fold = X_dev_norm[val_mask]
        y_val_fold = y_dev[val_mask]
        
        best_state, best_r2 = train_one_fold(
            X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, input_dim, output_dim, fold_i
        )
        fold_results.append(best_r2)
        
        save_path = os.path.join(BASE_DIR, WEIGHTS_DIR, f"{args.prefix}_fold{fold_i}.pth")
        torch.save({
            "state_dict": best_state,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": HIDDEN_SIZE,
            "n_lags": N_LAGS,
            "target_mode": args.target_mode
        }, save_path)
        
    print(f"\nMean CV R2: {np.mean(fold_results):.5f} +/- {np.std(fold_results):.5f}")
    
    # Pseudo-LB
    print("Evaluating on Pseudo-LB...")
    models = []
    for fold_i in range(CV_FOLDS):
        path = os.path.join(BASE_DIR, WEIGHTS_DIR, f"{args.prefix}_fold{fold_i}.pth")
        ckpt = torch.load(path)
        m = LagMLP(input_dim, HIDDEN_SIZE, output_dim)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        
    X_tens = torch.from_numpy(X_pseudo_norm)
    preds_accum = np.zeros_like(y_pseudo)
    with torch.no_grad():
        for m in models:
            preds_accum += m(X_tens).numpy()
    
    preds_ensemble = preds_accum / CV_FOLDS
    if args.target_mode == "residual":
        preds_level = preds_ensemble + prev_pseudo
        pseudo_r2s = [r2_score(y_pseudo_next[:, i], preds_level[:, i]) for i in range(output_dim)]
    else:
        pseudo_r2s = [r2_score(y_pseudo[:, i], preds_ensemble[:, i]) for i in range(output_dim)]
    print(f"Pseudo-LB Mean R2: {np.mean(pseudo_r2s):.5f}")

if __name__ == "__main__":
    main()
