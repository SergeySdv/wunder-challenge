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

# --- Refined TSMixer Classes (RevIN-Lite + Stem) ---

class RevINLite(nn.Module):
    def __init__(self, num_features, global_mean, global_std, alpha=0.5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.alpha = alpha
        self.affine = affine
        self.register_buffer('global_mean', torch.tensor(global_mean).float())
        self.register_buffer('global_std', torch.tensor(global_std).float())
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
            
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean_window = x.mean(dim=1, keepdim=True).detach()
            self.std_window = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            self.mean_eff = self.alpha * self.mean_window + (1 - self.alpha) * self.global_mean
            self.std_eff = self.alpha * self.std_window + (1 - self.alpha) * self.global_std
            x = (x - self.mean_eff) / self.std_eff
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            # Slice affine params to match output dim (32)
            out_dim = x.shape[-1]
            if self.affine:
                bias = self.affine_bias[:out_dim]
                weight = self.affine_weight[:out_dim]
                x = (x - bias) / (weight + 1e-5)
            # Slice stats
            mean_eff = self.mean_eff[..., :out_dim]
            std_eff = self.std_eff[..., :out_dim]
            x = x * std_eff + mean_eff
            return x

class TSMixerRefined(nn.Module):
    def __init__(self, input_channels, output_channels, seq_len, pred_len, global_mean, global_std, d_model=64, num_blocks=4, dropout=0.1, alpha=0.5):
        super().__init__()
        self.revin = RevINLite(input_channels, global_mean, global_std, alpha=alpha)
        self.stem = nn.Linear(input_channels, d_model)
        self.stem_act = nn.GELU()
        self.backbone = TSMixer(
            sequence_length=seq_len, prediction_length=pred_len,
            input_channels=d_model, output_channels=d_model,
            num_blocks=num_blocks, ff_dim=d_model * 2,
            dropout_rate=dropout, activation_fn="gelu",
            normalize_before=True, norm_type="batch"
        )
        self.head = nn.Linear(d_model, output_channels)
        
    def forward(self, x):
        x = self.revin(x, 'norm')
        x = self.stem(x)
        x = self.stem_act(x)
        x = self.backbone(x)
        x = self.head(x)
        x = self.revin(x, 'denorm')
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

# --- Main Prediction Model ---

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
            
        # --- Load TSMixer v5 Refined ---
        self.models = []
        norm_path = os.path.join(models_dir, "tsmixer_v5_refined_normalization.npz")
        norm = np.load(norm_path)
        
        self.clip_min = norm["clip_min"].astype(np.float32)
        self.clip_max = norm["clip_max"].astype(np.float32)
        global_mean = norm["global_mean"].astype(np.float32)
        global_std = norm["global_std"].astype(np.float32)
        alpha = float(norm["alpha"])
        
        self.n_lags = 10
        self.input_channels = 192
        
        for f in sorted([x for x in os.listdir(models_dir) if x.startswith("tsmixer_v5_refined_fold") and x.endswith(".pth")]):
            ckpt = torch.load(os.path.join(models_dir, f), map_location=self.device)
            cfg = ckpt.get("config", {})
            m = TSMixerRefined(
                input_channels=self.input_channels,
                output_channels=32,
                seq_len=self.n_lags,
                pred_len=1,
                global_mean=global_mean,
                global_std=global_std,
                d_model=cfg.get("d_model", 96),
                num_blocks=cfg.get("num_blocks", 4),
                dropout=cfg.get("dropout", 0.2),
                alpha=cfg.get("alpha", alpha)
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
            
        # Robust History Handling
        recent_hist = self.history[-20:] 
        df_hist = pd.DataFrame(np.array(recent_hist, dtype=np.float32))
        df_hist = df_hist.ffill().bfill().fillna(0.0)
        
        raw_window = df_hist.values[-10:]
        win = np.clip(raw_window, self.clip_min, self.clip_max)
        
        # Hybrid Feature Engineering
        diffs = np.zeros_like(win)
        diffs[1:] = win[1:] - win[:-1]
        
        mean_val = win.mean(axis=0)
        mean_block = np.tile(mean_val, (10, 1))
        
        std_val = win.std(axis=0)
        std_block = np.tile(std_val, (10, 1))
        
        slope_val = _compute_trend_slope(win)
        slope_block = np.tile(slope_val, (10, 1))
        
        denom = std_val + 1e-8
        standardized = (win - mean_val[None, :]) / denom[None, :]
        skew_val = (standardized**3).mean(axis=0)
        skew_block = np.tile(skew_val, (10, 1))
        
        # Concatenate: 32 + 32 + 32 + 32 + 32 + 32 = 192
        ts_input = np.concatenate([
            win, diffs, mean_block, std_block, slope_block, skew_block
        ], axis=1)
        
        # RevIN handles normalization, so we just cast
        t_ts = torch.from_numpy(ts_input).unsqueeze(0).float()
        
        pred = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            for m in self.models:
                pred += m(t_ts).squeeze().numpy()
        pred /= len(self.models)
        
        # Strict Clipping to Training Range
        # Use the min/max of the RAW input features (first 32 cols) as a proxy for reasonable output range
        pred = np.clip(pred, self.clip_min.min(), self.clip_max.max())
        
        return np.nan_to_num(pred, nan=0.0)

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import ScorerStepByStep
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'train.parquet')
    print("Scoring TSMixer v5 Refined locally...")
    scorer = ScorerStepByStep(dataset_path)
    res = scorer.score(PredictionModel())
    print(f"Mean R2: {res['mean_r2']:.6f}")
