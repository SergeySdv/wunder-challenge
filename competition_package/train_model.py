import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from typing import Tuple, List, Set, Optional, Dict

# --- Configuration ---
N_LAGS = 10
HIDDEN_SIZE = 256
N_EPOCHS = 20
BATCH_SIZE = 1024
LR = 1e-3
WEIGHTS_DIR = "models"
# We use a fixed seed for the Pseudo-LB split to ensure it stays constant across future experiments
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
    """
    Compute quantiles for the 32 raw feature columns.
    """
    feature_cols = [str(i) for i in range(32)]
    data = df[feature_cols].values
    lower = np.quantile(data, lower_q, axis=0).astype(np.float32)
    upper = np.quantile(data, upper_q, axis=0).astype(np.float32)
    return lower, upper

# --- Feature Engineering (Mirrored in solution.py) ---

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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build features X and target y.
    Applies WINSORIZATION (clipping) to the lag window inputs.
    Does NOT clip the targets y.
    """
    feature_cols = [str(i) for i in range(32)]
    
    X_list = []
    y_list = []
    
    # Group processing for speed
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
            
            # 1. Extract Raw Window
            raw_slice = states[idx - N_LAGS + 1 : idx + 1]
            
            # 2. Apply Winsorization (Clipping)
            # We clip the input window to remove extreme spikes before feature calc
            lag_slice = np.clip(raw_slice, clip_min, clip_max).astype(np.float32)
            
            # 3. Build Features (on clipped data)
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
            
            # Target: Raw next state (no clipping on target!)
            y_list.append(states[idx+1].astype(np.float32))
            
    return np.vstack(X_list), np.vstack(y_list)

# --- Model Architecture ---

class LagMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
            
        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
                targets.append(yb.numpy())
        
        preds = np.vstack(preds)
        targets = np.vstack(targets)
        
        # Mean R2
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
    print("--- v10 Robust Ensemble Experiment (Winsorization + 5-Fold CV) ---")
    
    # 1. Load Data
    dataset_path = os.path.join(BASE_DIR, "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    all_seqs = df["seq_ix"].unique()
    print(f"Total Sequences: {len(all_seqs)}")
    
    # 2. Pseudo-LB Split
    # We hold out ~10% of sequences completely to simulate the hidden test set.
    # These sequences are NOT used for calculating winsorization stats, normalization, or training.
    rng = np.random.default_rng(PSEUDO_LB_SEED) 
    rng.shuffle(all_seqs)
    
    n_pseudo = int(len(all_seqs) * 0.10)
    pseudo_lb_ids = all_seqs[:n_pseudo]
    dev_ids = all_seqs[n_pseudo:] # Used for CV (Train + Val)
    
    print(f"Pseudo-LB Size: {len(pseudo_lb_ids)} sequences (Held Out)")
    print(f"Dev Set Size:   {len(dev_ids)} sequences (For CV)")
    
    df_pseudo = df[df["seq_ix"].isin(pseudo_lb_ids)].copy()
    df_dev = df[df["seq_ix"].isin(dev_ids)].copy()
    
    # 3. Calculate Global Stats (Winsorization Bounds & Normalization) on DEV SET only
    print("Computing Winsorization bounds on Dev Set...")
    clip_min, clip_max = compute_winsorization_bounds(df_dev, 0.001, 0.999)
    
    print("Building dataset (applying winsorization) for Dev Set...")
    X_dev, y_dev = build_supervised_dataset(df_dev, clip_min, clip_max)
    
    print("Computing Normalization Stats on Dev Set...")
    x_mean = X_dev.mean(axis=0)
    x_std = X_dev.std(axis=0) + 1e-8
    
    # Save Meta-Parameters immediately (needed for solution.py)
    os.makedirs(os.path.join(BASE_DIR, WEIGHTS_DIR), exist_ok=True)
    norm_path = os.path.join(BASE_DIR, WEIGHTS_DIR, "lag_mlp_normalization.npz")
    np.savez(
        norm_path, 
        x_mean=x_mean, x_std=x_std, 
        clip_min=clip_min, clip_max=clip_max,
        n_lags=N_LAGS
    )
    print(f"Saved stats to {norm_path}")
    
    # Normalize Dev Data
    X_dev_norm = (X_dev - x_mean) / x_std
    
    # 4. 5-Fold Cross Validation Training
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    
    # We need to map back from X_dev indices to seq_ix to split correctly?
    # Actually, build_supervised_dataset flattened everything. 
    # We need to split by SEQ_IX, then rebuild or index. 
    # Better approach: Split dev_ids into 5 folds, then create boolean masks for X_dev.
    
    # To do this efficiently without rebuilding X every time:
    # We need a mapping from row_idx -> seq_ix.
    # Let's reconstruct the seq_ix column corresponding to X_dev rows.
    # Re-running build... just for metadata is fast enough? Or we can modify build to return seq_ix list.
    # Let's do a simpler way: We have dev_ids.
    
    fold_results = []
    
    input_dim = X_dev.shape[1]
    output_dim = y_dev.shape[1]
    
    print(f"Input Dim: {input_dim}, Output Dim: {output_dim}")
    
    # Create a map of seq_ix -> row_indices in X_dev
    # This requires us to know which seq_ix each row in X_dev belongs to.
    # Let's assume build_supervised_dataset iterates seqs in some order?
    # It iterates `df.groupby("seq_ix")`. The order of groups is sorted by key default in pandas groupby.
    # Let's verify we sorted the DF first? We did not sort df_dev by seq_ix explicitly before group, 
    # but groupby usually sorts. To be safe, let's rely on `df_dev.groupby("seq_ix")` order.
    
    # Re-generate seq_map to be 100% sure
    dev_seq_order = sorted(df_dev["seq_ix"].unique()) # Pandas groupby default sort
    
    # We need to know how many samples each sequence produced.
    seq_sample_counts = []
    for seq_ix, grp in df_dev.groupby("seq_ix"):
        # Logic must match build_supervised_dataset exactly
        # idx range: T-1 total steps.
        # conditions: need_pred[idx] AND idx >= N_LAGS-1
        grp = grp.sort_values("step_in_seq")
        need = grp["need_prediction"].values
        valid_mask = (need == 1) & (np.arange(len(need)) >= N_LAGS - 1) & (np.arange(len(need)) < len(need)-1) 
        # Last condition (idx < T-1) is implicit because loop goes to T-2.
        # Wait, loop is `range(T-1)`, so indices are 0..T-2.
        # The logic `if idx < N_LAGS - 1` handles the start.
        # The logic `if not need_pred[idx]` handles the mask.
        
        # Let's just count exactly as the builder does to be safe.
        count = 0
        T = len(grp)
        for idx in range(T-1):
            if need[idx] and idx >= N_LAGS - 1:
                count += 1
        seq_sample_counts.append(count)
        
    # Create array of seq_ix for every row in X_dev
    row_seq_ixs = []
    for s_ix, c in zip(dev_seq_order, seq_sample_counts):
        row_seq_ixs.extend([s_ix] * c)
    row_seq_ixs = np.array(row_seq_ixs)
    
    if len(row_seq_ixs) != len(X_dev):
        raise ValueError(f"Row count mismatch! X_dev: {len(X_dev)}, Map: {len(row_seq_ixs)}")
        
    print("Starting 5-Fold CV...")
    
    for fold_i, (train_idx_seq, val_idx_seq) in enumerate(kf.split(dev_ids)):
        # These indices index into `dev_ids`.
        train_seqs = dev_ids[train_idx_seq]
        val_seqs = dev_ids[val_idx_seq]
        
        # Boolean masks for X_dev rows
        train_mask = np.isin(row_seq_ixs, train_seqs)
        val_mask = np.isin(row_seq_ixs, val_seqs)
        
        X_tr_fold = X_dev_norm[train_mask]
        y_tr_fold = y_dev[train_mask]
        X_val_fold = X_dev_norm[val_mask]
        y_val_fold = y_dev[val_mask]
        
        print(f"Fold {fold_i}: Train Samples {len(X_tr_fold)}, Val Samples {len(X_val_fold)}")
        
        best_state, best_r2 = train_one_fold(
            X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, input_dim, output_dim, fold_i
        )
        
        fold_results.append(best_r2)
        
        # Save model
        save_path = os.path.join(BASE_DIR, WEIGHTS_DIR, f"lag_mlp_fold{fold_i}.pth")
        torch.save({
            "state_dict": best_state,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": HIDDEN_SIZE,
            "n_lags": N_LAGS
        }, save_path)
        
    print("\n--- CV Results ---")
    print(f"Fold R2s: {fold_results}")
    print(f"Mean CV R2: {np.mean(fold_results):.5f} +/- {np.std(fold_results):.5f}")
    
    # 5. Pseudo-LB Evaluation (Inference Mode)
    # We treat the pseudo-LB set exactly like test data:
    # Load the 5 models, use the global stats, predict, and score.
    print("\n--- Evaluating on Pseudo-LB (Held Out) ---")
    
    # Build Pseudo Features (Winsorized + Normalized)
    print("Building Pseudo-LB features...")
    X_pseudo, y_pseudo = build_supervised_dataset(df_pseudo, clip_min, clip_max)
    X_pseudo_norm = (X_pseudo - x_mean) / x_std
    
    # Load all 5 models for ensemble
    models = []
    for fold_i in range(CV_FOLDS):
        path = os.path.join(BASE_DIR, WEIGHTS_DIR, f"lag_mlp_fold{fold_i}.pth")
        ckpt = torch.load(path)
        m = LagMLP(input_dim, HIDDEN_SIZE, output_dim)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        
    # Predict
    X_tens = torch.from_numpy(X_pseudo_norm)
    preds_accum = np.zeros_like(y_pseudo)
    
    with torch.no_grad():
        for m in models:
            preds_accum += m(X_tens).numpy()
    
    preds_ensemble = preds_accum / CV_FOLDS
    
    # Score
    pseudo_r2s = [r2_score(y_pseudo[:, i], preds_ensemble[:, i]) for i in range(output_dim)]
    mean_pseudo_r2 = np.mean(pseudo_r2s)
    
    print(f"Pseudo-LB Mean R2: {mean_pseudo_r2:.5f}")
    
    # Save a text report
    with open(os.path.join(BASE_DIR, "EXPERIMENT_LOG.md"), "a") as f:
        f.write(f"\n\n## v10 Robust Ensemble (Winsorization + 5-Fold CV)\n")
        f.write(f"- Pseudo-LB Score: **{mean_pseudo_r2:.5f}** (Held out 10% seqs)\n")
        f.write(f"- CV Mean R2: **{np.mean(fold_results):.5f}** (Std: {np.std(fold_results):.5f})\n")
        f.write(f"- Strategy: Winsorize [0.1%, 99.9%] on inputs only. Global stats on Dev set.\n")

if __name__ == "__main__":
    main()