import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import r2_score

# Add project root folder to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from src.models.tsmixer_refined import TSMixerRefined

# --- Feature Engineering (Hybrid) ---
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
    N_LAGS = 10
    
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
    
    # Check if v5 exists
    norm_path = os.path.join(models_dir, "tsmixer_v5_refined_normalization.npz")
    if not os.path.exists(norm_path):
        print(f"Error: TSMixer v5 normalization not found at {norm_path}")
        print("Please run 'python scripts/train_tsmixer_v5_full.py' first.")
        return

    norm = np.load(norm_path)
    clip_min = norm["clip_min"].astype(np.float32)
    clip_max = norm["clip_max"].astype(np.float32)
    global_mean = norm["global_mean"].astype(np.float32)
    global_std = norm["global_std"].astype(np.float32)
    alpha = float(norm["alpha"])
    
    print("Building Hybrid dataset for TSMixer v5...")
    X_hard, y_hard = build_hybrid_dataset(df_hard, clip_min, clip_max)
    
    models = []
    device = torch.device("cpu")
    
    # Load TSMixer v5 models
    model_files = sorted([x for x in os.listdir(models_dir) if x.startswith("tsmixer_v5_refined_fold") and x.endswith(".pth")])
    if not model_files:
        print("Error: No TSMixer v5 model weights found.")
        return
        
    for f in model_files:
        path = os.path.join(models_dir, f)
        ckpt = torch.load(path, map_location=device)
        
        # Load config from checkpoint
        cfg = ckpt.get("config", {})
        n_lags = ckpt.get("n_lags", 10)
        
        m = TSMixerRefined(
            input_channels=X_hard.shape[2],
            output_channels=32,
            seq_len=n_lags,
            pred_len=1,
            global_mean=global_mean,
            global_std=global_std,
            d_model=cfg.get("d_model", 96),
            num_blocks=cfg.get("num_blocks", 4),
            dropout=cfg.get("dropout", 0.2),
            alpha=cfg.get("alpha", 0.5)
        )
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        
    print(f"Loaded {len(models)} TSMixer v5 models.")
    print("Evaluating TSMixer v5 on Hard Split...")
    
    X_tens = torch.from_numpy(X_hard).float()
    preds_accum = np.zeros_like(y_hard)
    
    # Batching for safety
    BATCH_SIZE = 512
    with torch.no_grad():
        for i in range(0, len(X_tens), BATCH_SIZE):
            xb = X_tens[i:i+BATCH_SIZE]
            batch_preds = np.zeros((len(xb), 32))
            for m in models:
                out = m(xb).squeeze(1).numpy()
                batch_preds += out
            preds_accum[i:i+BATCH_SIZE] = batch_preds
            
    preds = preds_accum / len(models)
    
    r2s = [r2_score(y_hard[:, i], preds[:, i]) for i in range(32)]
    mean_r2 = np.mean(r2s)
    
    print(f"TSMixer v5 Mean R2 on Hard Split: {mean_r2:.5f}")
    print(f"(Target to beat: MLP v19 ~0.35739)")

if __name__ == "__main__":
    main()
