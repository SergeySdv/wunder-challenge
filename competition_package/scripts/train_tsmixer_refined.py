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
NUM_BLOCKS = 4
D_MODEL = 96 # Tuned size
DROPOUT = 0.2
LR = 1e-3 # Higher for AdamW
ALPHA = 0.5 # 50% window, 50% global
DROP_PATH = 0.05
WARMUP_EPOCHS = 2
EMA_DECAY = 0.999
USE_Z_DELTA = True
REDUCE_CHANNELS = False  # If True, keep only lag/delta/mean/std blocks (128 ch)

N_LAGS = 10
BATCH_SIZE = 512
N_EPOCHS = 25
WEIGHTS_DIR = "models"
SAVE_PREFIX = "tsmixer_v5_refined"
SEED = 42

def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_blocks", type=int, default=NUM_BLOCKS)
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--drop_path", type=float, default=DROP_PATH)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--ema_decay", type=float, default=EMA_DECAY)
    parser.add_argument("--use_z_delta", action="store_true", default=USE_Z_DELTA)
    parser.add_argument("--no_z_delta", dest="use_z_delta", action="store_false")
    parser.add_argument("--reduce_channels", action="store_true", default=REDUCE_CHANNELS)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--save_prefix", type=str, default=SAVE_PREFIX)
    return parser.parse_args()

def compute_winsorization_bounds(df: pd.DataFrame, lower_q=0.001, upper_q=0.999):
    feature_cols = [str(i) for i in range(32)]
    data = df[feature_cols].values
    lower = np.quantile(data, lower_q, axis=0).astype(np.float32)
    upper = np.quantile(data, upper_q, axis=0).astype(np.float32)
    return lower, upper

# Reusing hybrid feature builder
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
            if USE_Z_DELTA:
                std_guard = np.clip(lag_slice.std(axis=0), 1e-4, None)
                diffs = diffs / std_guard[None, :]
            
            mean_val = lag_slice.mean(axis=0)
            mean_block = np.tile(mean_val, (N_LAGS, 1))
            
            std_val = lag_slice.std(axis=0)
            std_block = np.tile(std_val, (N_LAGS, 1))
            
            blocks = [lag_slice, diffs, mean_block, std_block]
            if not REDUCE_CHANNELS:
                slope_val = _compute_trend_slope(lag_slice)
                slope_block = np.tile(slope_val, (N_LAGS, 1))
                
                denom = std_val + 1e-8
                standardized = (lag_slice - mean_val[None, :]) / denom[None, :]
                skew_val = (standardized**3).mean(axis=0)
                skew_block = np.tile(skew_val, (N_LAGS, 1))
                blocks.extend([slope_block, skew_block])
            
            combined = np.concatenate(blocks, axis=1)
            
            X_list.append(combined)
            y_list.append(states[idx+1].astype(np.float32))
            
    return np.array(X_list), np.array(y_list)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[k] = v.detach().clone()
    def apply_to(self, model):
        model.load_state_dict(self.shadow)

def train_hard_split(X_train, y_train, X_val, y_val, global_mean, global_std):
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
        alpha=ALPHA,
        drop_path=DROP_PATH
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=max(1, WARMUP_EPOCHS)
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, N_EPOCHS - WARMUP_EPOCHS)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
    )
    loss_fn = nn.MSELoss()
    ema_helper = EMA(model, decay=EMA_DECAY)
    
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    best_val_r2 = -1e9
    best_state = None
    
    print(f"Training on {len(X_train)} samples, Validating on Hard Split ({len(X_val)} samples)...")
    
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(1)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            ema_helper.update(model)
            
        scheduler.step()
        
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            backup = model.state_dict()
            ema_helper.apply_to(model)
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).squeeze(1)
                preds.append(out.cpu().numpy())
                targets.append(yb.numpy())
            model.load_state_dict(backup)
        
        preds = np.vstack(preds)
        targets = np.vstack(targets)
        
        r2s = [r2_score(targets[:, i], preds[:, i]) for i in range(targets.shape[1])]
        mean_r2 = np.mean(r2s)
        
        if mean_r2 > best_val_r2:
            best_val_r2 = mean_r2
            best_state = {k: v.clone() for k, v in ema_helper.shadow.items()}
            
        print(f"  Epoch {epoch+1}/{N_EPOCHS} | Val R2: {mean_r2:.5f}")
            
    print(f"Best Hard-Val R2: {best_val_r2:.5f}")
    return best_state, best_val_r2

def main():
    args = parse_args()
    global NUM_BLOCKS, D_MODEL, DROPOUT, LR, ALPHA, DROP_PATH, WARMUP_EPOCHS, EMA_DECAY, USE_Z_DELTA, REDUCE_CHANNELS, N_EPOCHS, SAVE_PREFIX
    NUM_BLOCKS = args.num_blocks
    D_MODEL = args.d_model
    DROPOUT = args.dropout
    LR = args.lr
    ALPHA = args.alpha
    DROP_PATH = args.drop_path
    WARMUP_EPOCHS = args.warmup_epochs
    EMA_DECAY = args.ema_decay
    USE_Z_DELTA = args.use_z_delta
    REDUCE_CHANNELS = args.reduce_channels
    N_EPOCHS = args.n_epochs
    SAVE_PREFIX = args.save_prefix

    print(f"--- TSMixer v5 Refined (Hard Split Opt) ---")
    
    dataset_path = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    
    # Load Hard Split
    hard_split_path = os.path.join(CURRENT_DIR, "..", "datasets", "hard_validation_split.json")
    with open(hard_split_path, "r") as f:
        hard_seqs = json.load(f)
    
    all_seqs = df["seq_ix"].unique()
    train_seqs = [s for s in all_seqs if s not in hard_seqs]
    
    df_train = df[df["seq_ix"].isin(train_seqs)].copy()
    df_val = df[df["seq_ix"].isin(hard_seqs)].copy()
    
    print("Computing Winsorization bounds...")
    clip_min, clip_max = compute_winsorization_bounds(df_train)
    
    print("Building datasets...")
    X_train, y_train = build_hybrid_dataset(df_train, clip_min, clip_max)
    X_val, y_val = build_hybrid_dataset(df_val, clip_min, clip_max)
    
    # Compute Global Stats for RevIN
    # Flatten time and batch
    flat_X = X_train.reshape(-1, X_train.shape[2])
    global_mean = flat_X.mean(axis=0)
    global_std = flat_X.std(axis=0) + 1e-8
    
    os.makedirs(os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR), exist_ok=True)
    # Save config and stats
    norm_path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{SAVE_PREFIX}_normalization.npz")
    np.savez(norm_path, 
             clip_min=clip_min, clip_max=clip_max, 
             global_mean=global_mean, global_std=global_std,
             alpha=ALPHA)
    
    # Train
    best_state, best_r2 = train_hard_split(X_train, y_train, X_val, y_val, global_mean, global_std)
    
    # Save Best Model
    save_path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{SAVE_PREFIX}_hard.pth")
    torch.save({
        "state_dict": best_state,
        "config": {
            "d_model": D_MODEL,
            "num_blocks": NUM_BLOCKS,
            "dropout": DROPOUT,
            "alpha": ALPHA,
            "drop_path": DROP_PATH,
            "use_z_delta": USE_Z_DELTA,
            "reduce_channels": REDUCE_CHANNELS
        },
        "n_lags": N_LAGS
    }, save_path)

if __name__ == "__main__":
    main()
