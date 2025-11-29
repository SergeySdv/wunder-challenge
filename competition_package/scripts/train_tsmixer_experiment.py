import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import argparse

# Add project root folder to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from src.models.torchtsmixer import TSMixer

# Reuse data loading logic from train_model.py
# We need to copy some parts because we want RAW lags, not engineered features.
N_LAGS = 10
BATCH_SIZE = 512
WEIGHTS_DIR = "models"
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

def build_raw_dataset(
    df: pd.DataFrame, 
    clip_min: np.ndarray, 
    clip_max: np.ndarray
):
    """
    Builds dataset for TSMixer:
    X: (N, N_LAGS, 32)
    y: (N, 32)
    """
    feature_cols = [str(i) for i in range(32)]
    X_list = []
    y_list = []
    
    for seq_ix, df_seq in df.groupby("seq_ix"):
        df_seq = df_seq.sort_values("step_in_seq")
        states = df_seq[feature_cols].values
        need_pred = df_seq["need_prediction"].values
        
        T = len(df_seq)
        for idx in range(T - 1):
            if not need_pred[idx]:
                continue
            if idx < N_LAGS - 1:
                continue
            
            # Raw lag window
            raw_slice = states[idx - N_LAGS + 1 : idx + 1]
            # Winsorize
            lag_slice = np.clip(raw_slice, clip_min, clip_max).astype(np.float32)
            
            X_list.append(lag_slice)
            y_list.append(states[idx+1].astype(np.float32))
            
    return np.array(X_list), np.array(y_list)

def train_one_fold(X_train, y_train, X_val, y_val, args, fold_idx):
    device = torch.device("cpu")
    
    # TSMixer config
    # Updated to match torchtsmixer API
    model = TSMixer(
        sequence_length=N_LAGS,
        prediction_length=1,
        input_channels=32,
        output_channels=32,
        num_blocks=args.blocks,
        ff_dim=args.hidden,
        dropout_rate=args.dropout,
        activation_fn="relu",
        normalize_before=True,
        norm_type="batch"
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    loss_fn = nn.MSELoss()
    
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    best_val_r2 = -1e9
    best_state = None
    
    print(f"  [Fold {fold_idx}] Training TSMixer (blocks={args.blocks}, hidden={args.hidden})...")
    
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # Output is [batch, pred_len, channels], need to squeeze to [batch, channels]
            out = model(xb).squeeze(1) 
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            
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
        scheduler.step(mean_r2)
        
        if mean_r2 > best_val_r2:
            best_val_r2 = mean_r2
            best_state = model.state_dict()
            
    print(f"  [Fold {fold_idx}] Best Val R2: {best_val_r2:.5f}")
    return best_state, best_val_r2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_prefix", type=str, default="tsmixer_v1")
    args = parser.parse_args()
    
    dataset_path = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    
    all_seqs = df["seq_ix"].unique()
    rng = np.random.default_rng(PSEUDO_LB_SEED)
    rng.shuffle(all_seqs)
    
    n_pseudo = int(len(all_seqs) * 0.10)
    pseudo_ids = all_seqs[:n_pseudo]
    dev_ids = all_seqs[n_pseudo:]
    
    df_dev = df[df["seq_ix"].isin(dev_ids)].copy()
    df_pseudo = df[df["seq_ix"].isin(pseudo_ids)].copy()
    
    print("Computing Winsorization bounds on Dev set...")
    clip_min, clip_max = compute_winsorization_bounds(df_dev)
    
    print("Building raw datasets...")
    X_dev, y_dev = build_raw_dataset(df_dev, clip_min, clip_max)
    X_pseudo, y_pseudo = build_raw_dataset(df_pseudo, clip_min, clip_max)
    
    print(f"Dev shapes: X={X_dev.shape}, y={y_dev.shape}")
    
    # Normalize
    # We normalize per-feature across the whole dataset, similar to Lag-MLP
    # Input shape: (N, 10, 32) -> We compute stats over (N*10, 32) or (N, 32) from y?
    # Let's match Lag-MLP: compute global mean/std of the *features* from X_dev
    # Flatten time dim for stat computation
    X_dev_flat = X_dev.reshape(-1, 32)
    x_mean = X_dev_flat.mean(axis=0)
    x_std = X_dev_flat.std(axis=0) + 1e-8
    
    # Save normalization
    os.makedirs(os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR), exist_ok=True)
    norm_path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{args.save_prefix}_normalization.npz")
    np.savez(norm_path, x_mean=x_mean, x_std=x_std, clip_min=clip_min, clip_max=clip_max)
    
    # Apply normalization
    # (N, 10, 32) - (32,) broadcasts correctly on last dim
    X_dev_norm = (X_dev - x_mean) / x_std
    X_pseudo_norm = (X_pseudo - x_mean) / x_std
    
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    
    # Split logic (reuse simplistic splitting for now, ignoring rigorous sequence boundaries within fold for speed)
    # Actually, let's do it right: Split by SEQUENCE ID
    fold_results = []
    
    # Map sample index -> seq_ix for splitting
    # To do this accurately, we'd need to track seq_ix in build_raw_dataset.
    # For now, let's rely on the fact that we used df_dev. 
    # Let's just split dev_ids
    
    X_dev_norm_list = []
    y_dev_list = []
    seq_map = [] # store which seq_ix each sample belongs to
    
    # Re-build to keep track of sequences (a bit inefficient but safe)
    for seq_ix in dev_ids:
        df_seq = df_dev[df_dev["seq_ix"] == seq_ix]
        X_s, y_s = build_raw_dataset(df_seq, clip_min, clip_max)
        if len(X_s) > 0:
            X_norm_s = (X_s - x_mean) / x_std
            X_dev_norm_list.append(X_norm_s)
            y_dev_list.append(y_s)
            seq_map.extend([seq_ix] * len(X_s))
            
    X_dev_norm = np.concatenate(X_dev_norm_list)
    y_dev = np.concatenate(y_dev_list)
    seq_map = np.array(seq_map)
    
    print(f"Re-verified Dev shapes: X={X_dev_norm.shape}")
    
    for fold_i, (train_idx_seq, val_idx_seq) in enumerate(kf.split(dev_ids)):
        train_seqs = dev_ids[train_idx_seq]
        val_seqs = dev_ids[val_idx_seq]
        
        train_mask = np.isin(seq_map, train_seqs)
        val_mask = np.isin(seq_map, val_seqs)
        
        X_tr_fold = X_dev_norm[train_mask]
        y_tr_fold = y_dev[train_mask]
        X_val_fold = X_dev_norm[val_mask]
        y_val_fold = y_dev[val_mask]
        
        best_state, best_r2 = train_one_fold(
            X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, args, fold_i
        )
        fold_results.append(best_r2)
        
        save_path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{args.save_prefix}_fold{fold_i}.pth")
        torch.save({
            "state_dict": best_state,
            "config": vars(args),
            "n_lags": N_LAGS
        }, save_path)
        
    print(f"\nMean CV R2: {np.mean(fold_results):.5f}")
    
    # Pseudo-LB Eval
    print("Evaluating on Pseudo-LB...")
    models = []
    device = torch.device("cpu")
    for fold_i in range(CV_FOLDS):
        path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{args.save_prefix}_fold{fold_i}.pth")
        ckpt = torch.load(path)
        m = TSMixer(
            sequence_length=N_LAGS,
            prediction_length=1,
            input_channels=32,
            output_channels=32,
            num_blocks=args.blocks,
            ff_dim=args.hidden,
            dropout_rate=args.dropout,
            activation_fn="relu",
            normalize_before=True,
            norm_type="batch"
        )
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        
    X_pseudo_tens = torch.from_numpy(X_pseudo_norm)
    preds_accum = np.zeros_like(y_pseudo)
    
    with torch.no_grad():
        for m in models:
            out = m(X_pseudo_tens).squeeze(1).numpy()
            preds_accum += out
            
    preds_ensemble = preds_accum / CV_FOLDS
    pseudo_r2s = [r2_score(y_pseudo[:, i], preds_ensemble[:, i]) for i in range(32)]
    print(f"Pseudo-LB Mean R2: {np.mean(pseudo_r2s):.5f}")

if __name__ == "__main__":
    main()
