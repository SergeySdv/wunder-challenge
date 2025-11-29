import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

# --- TSMixer Classes ---
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

# --- Main Prediction Model (TSMixer Only) ---

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import DataPoint
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import DataPoint

class PredictionModel:
    def __init__(self):
        self.device = torch.device("cpu")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        if not os.path.exists(models_dir):
            models_dir = os.path.join(base_dir, "..", "models")
            
        self.models = []
        norm = np.load(os.path.join(models_dir, "tsmixer_v4_hybrid_normalization.npz"))
        self.mean = norm["x_mean"].astype(np.float32)
        self.std = norm["x_std"].astype(np.float32)
        self.clip_min = norm["clip_min"].astype(np.float32)
        self.clip_max = norm["clip_max"].astype(np.float32)
        
        self.n_lags = 10
        self.input_channels = 192
        
        for f in sorted([x for x in os.listdir(models_dir) if x.startswith("tsmixer_v4_hybrid_fold") and x.endswith(".pth")]):
            ckpt = torch.load(os.path.join(models_dir, f), map_location=self.device)
            m = TSMixer(
                sequence_length=self.n_lags,
                prediction_length=1,
                input_channels=self.input_channels,
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
            self.models.append(m)
            
        self.current_seq = None
        self.history = []
        torch.set_num_threads(1)
        
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq != data_point.seq_ix:
            self.current_seq = data_point.seq_ix
            self.history = []
            
        self.history.append(data_point.state)
        
        if not data_point.need_prediction:
            return None
            
        if len(self.history) < 10:
            return np.nan_to_num(data_point.state, nan=0.0)
            
        recent_hist = self.history[-20:] 
        df_hist = pd.DataFrame(np.array(recent_hist, dtype=np.float32))
        df_hist = df_hist.ffill().bfill().fillna(0.0)
        win_ts = df_hist.values[-10:]
        
        # Winsorize inputs
        win_ts = np.clip(win_ts, self.clip_min, self.clip_max)
        
        # Hybrid Feature Engineering
        diffs = np.zeros_like(win_ts)
        diffs[1:] = win_ts[1:] - win_ts[:-1]
        
        mean_val = win_ts.mean(axis=0)
        mean_block = np.tile(mean_val, (10, 1))
        
        std_val = win_ts.std(axis=0)
        std_block = np.tile(std_val, (10, 1))
        
        slope_val = _compute_trend_slope(win_ts)
        slope_block = np.tile(slope_val, (10, 1))
        
        denom = std_val + 1e-8
        standardized = (win_ts - mean_val[None, :]) / denom[None, :]
        skew_val = (standardized**3).mean(axis=0)
        skew_block = np.tile(skew_val, (10, 1))
        
        ts_input = np.concatenate([
            win_ts, diffs, mean_block, std_block, slope_block, skew_block
        ], axis=1)
        
        x_ts = (ts_input - self.mean) / self.std
        t_ts = torch.from_numpy(x_ts).unsqueeze(0).float()
        
        pred = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            for m in self.models:
                pred += m(t_ts).squeeze().numpy()
        pred /= len(self.models)
        
        return np.nan_to_num(pred, nan=0.0)

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import ScorerStepByStep
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'train.parquet')
    print("Scoring TSMixer v4 locally...")
    scorer = ScorerStepByStep(dataset_path)
    res = scorer.score(PredictionModel())
    print(f"Mean R2: {res['mean_r2']:.6f}")
