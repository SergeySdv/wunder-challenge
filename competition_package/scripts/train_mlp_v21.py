import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from typing import Tuple, List, Set, Optional, Dict
import argparse

# Adjust path to allow imports from src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from src.features.extractor import FeatureExtractor, feature_dim

# --- Configuration ---
N_LAGS = 10
HIDDEN_SIZE = 192
N_EPOCHS = 25
BATCH_SIZE = 512
LR = 1.6e-4
WEIGHTS_DIR = "models"
PSEUDO_LB_SEED = 999 
CV_FOLDS = 5
CV_SEED = 42
DROPOUT = 0.25  # Increased from 0.2 for v21

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

def build_supervised_dataset(
    df: pd.DataFrame, 
    extractor: FeatureExtractor,
    residual_target: bool = True
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
        # Use extractor's logic. But extractor is stateful.
        # Or we can use build_window_features directly on slices.
        # Let's use build_window_features on slices to be parallel-friendly if needed,
        # and consistent.
        
        # Extractor buffer logic is simple: [t-(N-1) ... t].
        for idx in range(T - 1):
            if not need_pred[idx]:
                continue
            if idx < N_LAGS - 1:
                continue
            
            # Slice for window: [idx - N_LAGS + 1, ..., idx] (inclusive of idx)
            # Length = N_LAGS
            raw_slice = states[idx - N_LAGS + 1 : idx + 1]
            
            # Extractor handles clipping
            features = extractor.build_window_features(raw_slice, steps[idx])
            
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
            nn.Dropout(DROPOUT),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
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
    parser = argparse.ArgumentParser(description="Train Lag-MLP v21 (Spreads + Residual).")
    parser.add_argument("--prefix", default="lag_mlp_v21", help="Prefix for saved model/normalization files.")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for testing.")
    args = parser.parse_args()

    print(f"--- v21 MLP (Spreads, Residual Targets) | prefix={args.prefix} ---")
    
    root_dir = os.path.abspath(os.path.join(BASE_DIR, ".."))
    dataset_path = os.path.join(root_dir, "datasets", "train.parquet")
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
    
    # Initialize Extractor with Spreads enabled
    extractor = FeatureExtractor(n_lags=N_LAGS, clip_min=clip_min, clip_max=clip_max, use_spreads=True)
    
    print("Building datasets...")
    X_dev, y_dev_next, prev_dev = build_supervised_dataset(df_dev, extractor)
    X_pseudo, y_pseudo_next, prev_pseudo = build_supervised_dataset(df_pseudo, extractor)

    # Residual Targets (v21 default)
    y_dev = y_dev_next - prev_dev
    y_pseudo = y_pseudo_next - prev_pseudo
    
    if args.subset:
        idx = np.random.choice(len(X_dev), args.subset, replace=False)
        X_dev = X_dev[idx]
        y_dev = y_dev[idx]
        prev_dev = prev_dev[idx] # Keep consistent for logic if needed later
    
    x_mean = X_dev.mean(axis=0)
    x_std = X_dev.std(axis=0) + 1e-8
    
    save_dir = os.path.join(root_dir, WEIGHTS_DIR)
    os.makedirs(save_dir, exist_ok=True)
    
    norm_path = os.path.join(save_dir, f"{args.prefix}_normalization.npz")
    np.savez(
        norm_path, 
        x_mean=x_mean, x_std=x_std, 
        clip_min=clip_min, clip_max=clip_max,
        n_lags=N_LAGS,
        use_spreads=True # Important metadata
    )
    
    X_dev_norm = (X_dev - x_mean) / x_std
    X_pseudo_norm = (X_pseudo - x_mean) / x_std
    
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    
    # Row sequence IDs for group splitting
    dev_seq_order = sorted(df_dev["seq_ix"].unique())
    seq_sample_counts = []
    # Re-scan to match X_dev rows to sequences (a bit expensive but safe)
    # NOTE: build_supervised_dataset appends sequentially by seq_ix group.
    # If subset is used, this logic breaks. But subset is for debugging.
    # For full run:
    if not args.subset:
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
    else:
        # Dummy splitting for subset debugging
        row_seq_ixs = np.arange(len(X_dev)) # random split effectively
    
    fold_results = []
    input_dim = X_dev.shape[1]
    output_dim = y_dev.shape[1]
    
    print(f"Feature Dim: {input_dim} (Expected: {feature_dim(N_LAGS, True)})")
    
    print(f"Starting {CV_FOLDS}-Fold CV...")
    for fold_i, (train_idx_seq, val_idx_seq) in enumerate(kf.split(dev_ids)):
        if not args.subset:
            train_seqs = dev_ids[train_idx_seq]
            val_seqs = dev_ids[val_idx_seq]
            
            train_mask = np.isin(row_seq_ixs, train_seqs)
            val_mask = np.isin(row_seq_ixs, val_seqs)
            
            X_tr_fold = X_dev_norm[train_mask]
            y_tr_fold = y_dev[train_mask]
            X_val_fold = X_dev_norm[val_mask]
            y_val_fold = y_dev[val_mask]
        else:
            # Random KFold on indices for subset
            kf_sub = KFold(n_splits=CV_FOLDS)
            tr, val = list(kf_sub.split(X_dev))[fold_i]
            X_tr_fold, y_tr_fold = X_dev_norm[tr], y_dev[tr]
            X_val_fold, y_val_fold = X_dev_norm[val], y_dev[val]
        
        best_state, best_r2 = train_one_fold(
            X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, input_dim, output_dim, fold_i
        )
        fold_results.append(best_r2)
        
        save_path = os.path.join(save_dir, f"{args.prefix}_fold{fold_i}.pth")
        torch.save({
            "state_dict": best_state,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": HIDDEN_SIZE,
            "n_lags": N_LAGS,
            "use_spreads": True
        }, save_path)
        
    print(f"\nMean CV R2 (Residual): {np.mean(fold_results):.5f} +/- {np.std(fold_results):.5f}")
    
    # Pseudo-LB
    print("Evaluating on Pseudo-LB...")
    models = []
    for fold_i in range(CV_FOLDS):
        path = os.path.join(save_dir, f"{args.prefix}_fold{fold_i}.pth")
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
    
    preds_ensemble_residual = preds_accum / CV_FOLDS
    # Reconstruct level
    preds_level = preds_ensemble_residual + prev_pseudo
    
    pseudo_r2s = [r2_score(y_pseudo_next[:, i], preds_level[:, i]) for i in range(output_dim)]
    print(f"Pseudo-LB Mean R2 (Level): {np.mean(pseudo_r2s):.5f}")

if __name__ == "__main__":
    main()
