import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Add project root folder to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from src.models.tsmixer_refined import TSMixerRefined

# --- Config ---
# Full Training Configuration
NUM_BLOCKS = 4
D_MODEL = 96 
DROPOUT = 0.2
LR = 1e-3 
ALPHA = 0.5

N_LAGS = 10
BATCH_SIZE = 512
N_EPOCHS = 25 # Full epochs
WEIGHTS_DIR = "models"
SAVE_PREFIX = "tsmixer_v5_refined"
CV_FOLDS = 5
CV_SEED = 42
PSEUDO_LB_SEED = 999

def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    return df

def compute_winsorization_bounds(df: pd.DataFrame, lower_q=0.001, upper_q=0.999):
    feature_cols = [str(i) for i in range(32)]
    data = df[feature_cols].values
    lower = np.quantile(data, lower_q, axis=0).astype(np.float32)
    upper = np.quantile(data, upper_q, axis=0).astype(np.float32)
    return lower, upper

def _compute_trend_slope(lag_slice):
    n_lags = lag_slice.shape[0]
    t = np.arange(n_lags, dtype=np.float32)
    sum_t = float(n_lags * (n_lags - 1) / 2.0)
    sum_t2 = float(n_lags * (n_lags - 1) * (2 * n_lags - 1) / 6.0)
    sum_y = lag_slice.sum(axis=0)
    sum_ty = (t[:, None] * lag_slice).sum(axis=0)
    denom = n_lags * sum_t2 - sum_t * sum_t
    slope = np.zeros(lag_slice.shape[1], dtype=np.float32)
    if denom != 0.0:
        slope = (n_lags * sum_ty - sum_t * sum_y) / denom
    return slope

def build_hybrid_dataset(df: pd.DataFrame, clip_min: np.ndarray, clip_max: np.ndarray):
    feature_cols = [str(i) for i in range(32)]
    X_list = []
    y_list = []
    
    for seq_ix, df_seq in df.groupby("seq_ix"):
        df_seq = df_seq.sort_values("step_in_seq")
        states = df_seq[feature_cols].values
        need_pred = df_seq["need_prediction"].values
        
        T = len(df_seq)
        for idx in range(T - 1):
            if not need_pred[idx]: continue
            if idx < N_LAGS - 1: continue
            
            raw_slice = states[idx - N_LAGS + 1 : idx + 1]
            lag_slice = np.clip(raw_slice, clip_min, clip_max).astype(np.float32)
            
            diffs = np.zeros_like(lag_slice)
            diffs[1:] = lag_slice[1:] - lag_slice[:-1]
            
            mean_val = lag_slice.mean(axis=0)
            mean_block = np.tile(mean_val, (N_LAGS, 1))
            
            std_val = lag_slice.std(axis=0)
            std_block = np.tile(std_val, (N_LAGS, 1))
            
            slope_val = _compute_trend_slope(lag_slice)
            slope_block = np.tile(slope_val, (N_LAGS, 1))
            
            denom = std_val + 1e-8
            standardized = (lag_slice - mean_val[None, :]) / denom[None, :]
            skew_val = (standardized**3).mean(axis=0)
            skew_block = np.tile(skew_val, (N_LAGS, 1))
            
            combined = np.concatenate([
                lag_slice, diffs, mean_block, std_block, slope_block, skew_block
            ], axis=1)
            
            X_list.append(combined)
            y_list.append(states[idx+1].astype(np.float32))
            
    return np.array(X_list), np.array(y_list)

def train_one_fold(X_train, y_train, X_val, y_val, fold_idx, global_mean, global_std):
    device = torch.device("cpu")
    input_channels = X_train.shape[2]
    
    model = TSMixerRefined(
        input_channels=input_channels,
        output_channels=32,
        seq_len=N_LAGS,
        pred_len=1,
        global_mean=global_mean,
        global_std=global_std,
        d_model=D_MODEL,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
        alpha=ALPHA
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    loss_fn = nn.MSELoss()
    
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    best_val_r2 = -1e9
    best_state = None
    
    print(f"  [Fold {fold_idx}] Training...")
    
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(1)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).squeeze(1)
                preds.append(out.cpu().numpy())
                targets.append(yb.numpy())
        
        preds = np.vstack(preds)
        targets = np.vstack(targets)
        
        r2s = [r2_score(targets[:, i], preds[:, i]) for i in range(targets.shape[1])]
        mean_r2 = np.mean(r2s)
        
        if mean_r2 > best_val_r2:
            best_val_r2 = mean_r2
            best_state = model.state_dict()
            
    print(f"  [Fold {fold_idx}] Best Val R2: {best_val_r2:.5f}")
    return best_state, best_val_r2

def main():
    print(f"--- TSMixer v5 Refined (Full Training on Hard Split) ---")
    
    dataset_path = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    
    # Load Hard Split (Strategic Pivot)
    hard_split_path = os.path.join(CURRENT_DIR, "..", "datasets", "hard_validation_split.json")
    if os.path.exists(hard_split_path):
        print(f"Loading Hard Split from {hard_split_path}...")
        with open(hard_split_path, "r") as f:
            hard_seqs = json.load(f)
        
        # Pseudo-LB = Hard Split
        pseudo_ids = np.array(hard_seqs)
        all_seqs = df["seq_ix"].unique()
        dev_ids = np.array([s for s in all_seqs if s not in pseudo_ids])
        
        print(f"Train Seqs: {len(dev_ids)}, Hard-Val Seqs: {len(pseudo_ids)}")
    else:
        print("Hard split not found! Falling back to random split.")
        all_seqs = df["seq_ix"].unique()
        rng = np.random.default_rng(PSEUDO_LB_SEED)
        rng.shuffle(all_seqs)
        
        n_pseudo = int(len(all_seqs) * 0.10)
        pseudo_ids = all_seqs[:n_pseudo]
        dev_ids = all_seqs[n_pseudo:]
    
    df_dev = df[df["seq_ix"].isin(dev_ids)].copy()
    df_pseudo = df[df["seq_ix"].isin(pseudo_ids)].copy()
    
    print("Computing Winsorization bounds...")
    clip_min, clip_max = compute_winsorization_bounds(df_dev)
    
    print("Building datasets...")
    X_dev, y_dev = build_hybrid_dataset(df_dev, clip_min, clip_max)
    X_pseudo, y_pseudo = build_hybrid_dataset(df_pseudo, clip_min, clip_max)
    
    # Compute Global Stats for RevIN
    flat_X = X_dev.reshape(-1, X_dev.shape[2])
    global_mean = flat_X.mean(axis=0)
    global_std = flat_X.std(axis=0) + 1e-8
    
    os.makedirs(os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR), exist_ok=True)
    norm_path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{SAVE_PREFIX}_normalization.npz")
    np.savez(norm_path, 
             clip_min=clip_min, clip_max=clip_max, 
             global_mean=global_mean, global_std=global_std,
             alpha=ALPHA)
             
    # Reconstruction for splitting
    sorted_seqs = sorted(dev_ids)
    seq_counts = [899] * len(sorted_seqs) 
    seq_map = np.repeat(sorted_seqs, seq_counts)
    if len(seq_map) > len(X_dev): seq_map = seq_map[:len(X_dev)]
    
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_results = []
    
    for fold_i, (train_idx_seq, val_idx_seq) in enumerate(kf.split(sorted_seqs)):
        train_seqs = np.array(sorted_seqs)[train_idx_seq]
        val_seqs = np.array(sorted_seqs)[val_idx_seq]
        
        train_mask = np.isin(seq_map, train_seqs)
        val_mask = np.isin(seq_map, val_seqs)
        
        X_tr_fold = X_dev[train_mask]
        y_tr_fold = y_dev[train_mask]
        X_val_fold = X_dev[val_mask]
        y_val_fold = y_dev[val_mask]
        
        best_state, best_r2 = train_one_fold(
            X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, fold_i,
            global_mean, global_std
        )
        fold_results.append(best_r2)
        
        save_path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{SAVE_PREFIX}_fold{fold_i}.pth")
        torch.save({
            "state_dict": best_state,
            "config": {
                "d_model": D_MODEL,
                "num_blocks": NUM_BLOCKS,
                "dropout": DROPOUT,
                "alpha": ALPHA
            },
            "n_lags": N_LAGS
        }, save_path)
        
    print(f"\nMean CV R2: {np.mean(fold_results):.5f}")
    
    # Pseudo-LB
    print("Evaluating on Pseudo-LB...")
    models = []
    device = torch.device("cpu")
    for fold_i in range(CV_FOLDS):
        path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{SAVE_PREFIX}_fold{fold_i}.pth")
        ckpt = torch.load(path)
        input_channels = X_dev.shape[2]
        m = TSMixerRefined(
            input_channels=input_channels,
            output_channels=32,
            seq_len=N_LAGS,
            pred_len=1,
            global_mean=global_mean,
            global_std=global_std,
            d_model=D_MODEL,
            num_blocks=NUM_BLOCKS,
            dropout=DROPOUT,
            alpha=ALPHA
        )
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        
    X_pseudo_tens = torch.from_numpy(X_pseudo)
    preds_accum = np.zeros_like(y_pseudo)
    
    with torch.no_grad():
        for i in range(0, len(X_pseudo_tens), BATCH_SIZE):
            batch = X_pseudo_tens[i:i+BATCH_SIZE]
            batch_preds = np.zeros((len(batch), 32))
            for m in models:
                out = m(batch).squeeze(1).numpy()
                batch_preds += out
            preds_accum[i:i+BATCH_SIZE] = batch_preds
            
    preds_ensemble = preds_accum / CV_FOLDS
    pseudo_r2s = [r2_score(y_pseudo[:, i], preds_ensemble[:, i]) for i in range(32)]
    print(f"Pseudo-LB Mean R2: {np.mean(pseudo_r2s):.5f}")

if __name__ == "__main__":
    main()
