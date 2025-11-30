import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import r2_score

# Add project root folder to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

# --- TSMixer Classes (Inline for standalone script) ---
class TimeBatchNorm2d(nn.BatchNorm1d):
    def __init__(self, normalized_shape: tuple):
        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels
    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1, 1)
        x = super().forward(x)
        x = x.reshape(x.shape[0], self.num_time_steps, self.num_channels)
        return x

class FeatureMixing(nn.Module):
    def __init__(self, sequence_length, input_channels, output_channels, ff_dim, activation_fn, dropout_rate, normalize_before, norm_type):
        super().__init__()
        self.norm_before = norm_type((sequence_length, input_channels)) if normalize_before else nn.Identity()
        self.norm_after = norm_type((sequence_length, output_channels)) if not normalize_before else nn.Identity()
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_channels)
        self.projection = nn.Linear(input_channels, output_channels) if input_channels != output_channels else nn.Identity()
    def forward(self, x):
        x_proj = self.projection(x)
        x = self.norm_before(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x_proj + x
        return self.norm_after(x)

class TimeMixing(nn.Module):
    def __init__(self, sequence_length, input_channels, activation_fn, dropout_rate, norm_type):
        super().__init__()
        self.norm = norm_type((sequence_length, input_channels))
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(sequence_length, sequence_length)
    def forward(self, x):
        x_temp = x.permute(0, 2, 1)
        x_temp = self.activation_fn(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        x_res = x_temp.permute(0, 2, 1)
        return self.norm(x + x_res)

class MixerLayer(nn.Module):
    def __init__(self, sequence_length, input_channels, output_channels, ff_dim, activation_fn, dropout_rate, normalize_before, norm_type):
        super().__init__()
        self.time_mixing = TimeMixing(sequence_length, input_channels, activation_fn, dropout_rate, norm_type)
        self.feature_mixing = FeatureMixing(sequence_length, input_channels, output_channels, ff_dim, activation_fn, dropout_rate, normalize_before, norm_type)
    def forward(self, x):
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x

class TSMixer(nn.Module):
    def __init__(self, sequence_length, prediction_length, input_channels, output_channels=None, activation_fn="relu", num_blocks=2, dropout_rate=0.1, ff_dim=64, normalize_before=True, norm_type="batch"):
        super().__init__()
        activation_fn = getattr(F, activation_fn)
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm
        output_channels = output_channels if output_channels is not None else input_channels
        channels = [input_channels] * (num_blocks - 1) + [output_channels]
        self.mixer_layers = nn.Sequential(*[
            MixerLayer(sequence_length, in_ch, out_ch, ff_dim, activation_fn, dropout_rate, normalize_before, norm_type)
            for in_ch, out_ch in zip(channels[:-1], channels[1:])
        ])
        self.temporal_projection = nn.Linear(sequence_length, prediction_length)
    def forward(self, x_hist):
        x = self.mixer_layers(x_hist)
        x_temp = x.permute(0, 2, 1)
        x_temp = self.temporal_projection(x_temp)
        x = x_temp.permute(0, 2, 1)
        return x

# --- Feature Engineering ---
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
    norm = np.load(os.path.join(models_dir, "tsmixer_v4_hybrid_normalization.npz"))
    x_mean = norm["x_mean"].astype(np.float32)
    x_std = norm["x_std"].astype(np.float32)
    clip_min = norm["clip_min"].astype(np.float32)
    clip_max = norm["clip_max"].astype(np.float32)
    
    print("Building Hybrid dataset for TSMixer...")
    X_hard, y_hard = build_hybrid_dataset(df_hard, clip_min, clip_max)
    
    # Broadcast normalization logic from training script
    flat_X = X_hard.reshape(-1, 192)
    # Note: In training we flattened (N*10, 192) to compute stats.
    # Here x_mean/std shape is (192,).
    # X_hard is (N, 10, 192).
    X_norm = (X_hard - x_mean) / x_std
    
    models = []
    device = torch.device("cpu")
    # Load TSMixer v4 models
    for f in sorted([x for x in os.listdir(models_dir) if x.startswith("tsmixer_v4_hybrid_fold") and x.endswith(".pth")]):
        ckpt = torch.load(os.path.join(models_dir, f), map_location=device)
        m = TSMixer(
            sequence_length=10,
            prediction_length=1,
            input_channels=192,
            output_channels=32,
            num_blocks=2,
            ff_dim=256,
            dropout_rate=0.2,
            activation_fn="relu",
            normalize_before=True,
            norm_type="batch"
        )
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        
    print(f"Loaded {len(models)} TSMixer models.")
    print("Evaluating TSMixer v4 on Hard Split...")
    X_tens = torch.from_numpy(X_norm).float()
    preds_accum = np.zeros_like(y_hard)
    
    with torch.no_grad():
        for m in models:
            preds_accum += m(X_tens).squeeze().numpy()
    preds = preds_accum / len(models)
    
    r2s = [r2_score(y_hard[:, i], preds[:, i]) for i in range(32)]
    mean_r2 = np.mean(r2s)
    
    print(f"TSMixer v4 Mean R2 on Hard Split: {mean_r2:.5f}")

if __name__ == "__main__":
    main()
