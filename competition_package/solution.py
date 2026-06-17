import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

# --- Utility ---
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

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
            out_dim = x.shape[-1]
            if self.affine:
                bias = self.affine_bias[:out_dim]
                weight = self.affine_weight[:out_dim]
                x = (x - bias) / (weight + 1e-5)
            mean_eff = self.mean_eff[..., :out_dim]
            std_eff = self.std_eff[..., :out_dim]
            x = x * std_eff + mean_eff
            return x

class TSMixerRefined(nn.Module):
    def __init__(self, input_channels, output_channels, seq_len, pred_len, global_mean, global_std, d_model=64, num_blocks=4, dropout=0.1, alpha=0.5, drop_path: float = 0.0):
        super().__init__()
        self.revin = RevINLite(input_channels, global_mean, global_std, alpha=alpha)
        self.drop_path_rate = drop_path
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
        x = drop_path(x, self.drop_path_rate, self.training)
        x = self.head(x)
        x = self.revin(x, 'denorm')
        return x

# --- MLP Classes ---
class LagMLP_v19(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x): return self.net(x)

class LagMLP_v21(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x): return self.net(x)

# --- Helper Functions ---
def _compute_lag1_autocorr(lag_slice):
    x = lag_slice[:-1, :]
    y = lag_slice[1:, :]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    num = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    denom = np.sqrt(((x - x_mean)**2).mean(axis=0) * ((y - y_mean)**2).mean(axis=0)) + 1e-8
    return num / denom

def _compute_lagk_autocorr(lag_slice, lag):
    if lag <= 0 or lag >= lag_slice.shape[0]: return np.zeros(lag_slice.shape[1], dtype=np.float32)
    x = lag_slice[:-lag, :]
    y = lag_slice[lag:, :]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    num = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    denom = np.sqrt(((x - x_mean)**2).mean(axis=0) * ((y - y_mean)**2).mean(axis=0)) + 1e-8
    return num / denom

def _compute_frac_above_mean(lag_slice, mean_last):
    return (lag_slice > mean_last[None, :]).mean(axis=0)

def _compute_robust_window_stats(lag_slice, mean_last, std_last):
    q25 = np.percentile(lag_slice, 25, axis=0)
    median = np.percentile(lag_slice, 50, axis=0)
    q75 = np.percentile(lag_slice, 75, axis=0)
    iqr = q75 - q25
    denom = std_last + 1e-8
    standardized = (lag_slice - mean_last[None, :]) / denom[None, :]
    skewness = (standardized**3).mean(axis=0)
    kurtosis = ((standardized**4).mean(axis=0) - 3.0)
    cv = std_last / (np.abs(mean_last) + 1e-8)
    return q25, median, q75, iqr, skewness, kurtosis, cv

def _compute_trend_features(lag_slice):
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
    mid = n_lags // 2
    slope_first = (lag_slice[mid, :] - lag_slice[0, :]) / float(mid)
    slope_second = (lag_slice[-1, :] - lag_slice[mid, :]) / float(n_lags - mid)
    curvature = slope_second - slope_first
    return slope, r2, curvature

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
            
        # --- 1. Load MLP v19 (Level) ---
        self.mlp_v19_models = []
        norm_v19 = np.load(os.path.join(models_dir, "lag_mlp_normalization.npz"))
        self.mean_v19 = norm_v19["x_mean"].astype(np.float32)
        self.std_v19 = norm_v19["x_std"].astype(np.float32)
        self.clip_min_v19 = norm_v19["clip_min"].astype(np.float32)
        self.clip_max_v19 = norm_v19["clip_max"].astype(np.float32)
        
        for f in sorted([x for x in os.listdir(models_dir) if x.startswith("lag_mlp_fold") and x.endswith("_fp16.pth")]):
            ckpt = torch.load(os.path.join(models_dir, f), map_location=self.device)
            # De-quantize state dict to float32 for inference
            for k, v in ckpt["state_dict"].items():
                ckpt["state_dict"][k] = v.float()
                
            m = LagMLP_v19(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["output_dim"])
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.mlp_v19_models.append(m)
        
        # --- 2. Load MLP v21 (Residual) ---
        self.mlp_v21_models = []
        norm_v21 = np.load(os.path.join(models_dir, "lag_mlp_v21_normalization.npz"))
        self.mean_v21 = norm_v21["x_mean"].astype(np.float32)
        self.std_v21 = norm_v21["x_std"].astype(np.float32)
        self.clip_min_v21 = norm_v21["clip_min"].astype(np.float32)
        self.clip_max_v21 = norm_v21["clip_max"].astype(np.float32)
        
        for f in sorted([x for x in os.listdir(models_dir) if x.startswith("lag_mlp_residual_fold") and x.endswith("_fp16.pth")]):
            ckpt = torch.load(os.path.join(models_dir, f), map_location=self.device)
            for k, v in ckpt["state_dict"].items():
                ckpt["state_dict"][k] = v.float()
                
            m = LagMLP_v21(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["output_dim"])
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.mlp_v21_models.append(m)
        
        # --- 3. Load TSMixer v5 Refined (RevIN) ---
        self.tsmixer_models = []
        norm_ts = np.load(os.path.join(models_dir, "tsmixer_v5_refined_normalization.npz"))
        self.clip_min_ts = norm_ts["clip_min"].astype(np.float32)
        self.clip_max_ts = norm_ts["clip_max"].astype(np.float32)
        
        global_mean = norm_ts["global_mean"].astype(np.float32)
        global_std = norm_ts["global_std"].astype(np.float32)
        alpha = float(norm_ts["alpha"])
        
        self.ts_n_lags = 10
        self.ts_input_channels = 192
        self.use_z_delta_ts = True
        self.reduce_channels_ts = False
        
        for f in sorted([x for x in os.listdir(models_dir) if x.startswith("tsmixer_v5_refined_fold") and x.endswith("_fp16.pth")]):
            ckpt = torch.load(os.path.join(models_dir, f), map_location=self.device)
            for k, v in ckpt["state_dict"].items():
                ckpt["state_dict"][k] = v.float()
                
            cfg = ckpt.get("config", {})
            self.use_z_delta_ts = cfg.get("use_z_delta", True)
            self.reduce_channels_ts = cfg.get("reduce_channels", False)
            self.ts_input_channels = 128 if self.reduce_channels_ts else 192
            m = TSMixerRefined(
                input_channels=self.ts_input_channels,
                output_channels=32,
                seq_len=self.ts_n_lags,
                pred_len=1,
                global_mean=global_mean,
                global_std=global_std,
                d_model=cfg.get("d_model", 96),
                num_blocks=cfg.get("num_blocks", 4),
                dropout=cfg.get("dropout", 0.2),
                alpha=cfg.get("alpha", alpha),
                drop_path=cfg.get("drop_path", 0.0)
            )
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.tsmixer_models.append(m)
            
        self.current_seq = None
        self.history = []
        torch.set_num_threads(1)
        
    def _build_mlp_features(self, lag_slice, use_spreads=False):
        lag_flat = lag_slice.reshape(-1)
        last = lag_slice[-1]
        delta_flat = (lag_slice - last).reshape(-1)
        mean_last = lag_slice.mean(axis=0)
        std_last = lag_slice.std(axis=0)
        ac1 = _compute_lag1_autocorr(lag_slice)
        ac2 = _compute_lagk_autocorr(lag_slice, 2)
        ac3 = _compute_lagk_autocorr(lag_slice, 3)
        acf_sum = np.abs(ac1) + np.abs(ac2) + np.abs(ac3)
        frac = _compute_frac_above_mean(lag_slice, mean_last)
        q25, med, q75, iqr, skew, kurt, cv = _compute_robust_window_stats(lag_slice, mean_last, std_last)
        slope, r2, curve = _compute_trend_features(lag_slice)
        
        spreads = []
        if use_spreads:
            pairs = [(18, 28), (1, 28)]
            for i, j in pairs:
                spreads.append(last[i] - last[j])
            spreads = np.array(spreads, dtype=np.float32)
            
        features = np.concatenate([
            lag_flat, delta_flat, mean_last, std_last,
            ac1, ac2, ac3, acf_sum, frac,
            q25, med, q75, iqr, skew, kurt, cv,
            slope, r2, curve,
            spreads if use_spreads else []
        ])
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq != data_point.seq_ix:
            self.current_seq = data_point.seq_ix
            self.history = []
            
        self.history.append(data_point.state)
        
        if not data_point.need_prediction:
            return None
            
        if len(self.history) < 10:
            return np.nan_to_num(data_point.state, nan=0.0)
            
        # Robust Pandas Filling
        recent_hist = self.history[-20:] 
        df_hist = pd.DataFrame(np.array(recent_hist, dtype=np.float32))
        df_hist = df_hist.ffill().bfill().fillna(0.0)
        raw_window = df_hist.values[-10:]
        
        step_val = np.array([data_point.step_in_seq / 1000.0], dtype=np.float32)
        
        # --- 1. MLP v19 ---
        win_v19 = np.clip(raw_window, self.clip_min_v19, self.clip_max_v19)
        feats_v19 = self._build_mlp_features(win_v19, use_spreads=False)
        feats_v19 = np.concatenate([feats_v19, step_val])
        x_v19 = (feats_v19 - self.mean_v19) / self.std_v19
        t_v19 = torch.from_numpy(x_v19).unsqueeze(0).float()
        
        pred_v19 = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            for m in self.mlp_v19_models:
                pred_v19 += m(t_v19).squeeze().numpy()
        pred_v19 /= len(self.mlp_v19_models)
        pred_v19 = np.clip(pred_v19, self.clip_min_v19.min(), self.clip_max_v19.max())
        
        # --- 2. MLP v21 (Residual target) ---
        win_v21 = np.clip(raw_window, self.clip_min_v21, self.clip_max_v21)
        feats_v21 = self._build_mlp_features(win_v21, use_spreads=False)
        feats_v21 = np.concatenate([feats_v21, step_val])
        if self.mean_v21.shape[0] != feats_v21.shape[0]:
            mean_v21 = self.mean_v21[:feats_v21.shape[0]]
            std_v21 = self.std_v21[:feats_v21.shape[0]]
        else:
            mean_v21 = self.mean_v21
            std_v21 = self.std_v21
        x_v21 = (feats_v21 - mean_v21) / std_v21
        t_v21 = torch.from_numpy(x_v21).unsqueeze(0).float()
        
        pred_v21_resid = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            for m in self.mlp_v21_models:
                pred_v21_resid += m(t_v21).squeeze().numpy()
        pred_v21_resid /= len(self.mlp_v21_models)
        pred_v21 = win_v21[-1] + pred_v21_resid
        pred_v21 = np.clip(pred_v21, self.clip_min_v21.min(), self.clip_max_v21.max())
        
        # --- 3. TSMixer v5 Refined ---
        win_ts = np.clip(raw_window, self.clip_min_ts, self.clip_max_ts)
        diffs = np.zeros_like(win_ts)
        diffs[1:] = win_ts[1:] - win_ts[:-1]
        if self.use_z_delta_ts:
            std_guard = np.clip(win_ts.std(axis=0), 1e-4, None)
            diffs = diffs / std_guard[None, :]
        
        mean_val = win_ts.mean(axis=0)
        mean_block = np.tile(mean_val, (10, 1))
        std_val = win_ts.std(axis=0)
        std_block = np.tile(std_val, (10, 1))
        
        ts_blocks = [win_ts, diffs, mean_block, std_block]
        if not self.reduce_channels_ts:
            slope_val = _compute_trend_slope(win_ts)
            slope_block = np.tile(slope_val, (10, 1))
            denom = std_val + 1e-8
            standardized = (win_ts - mean_val[None, :]) / denom[None, :]
            skew_val = (standardized**3).mean(axis=0)
            skew_block = np.tile(skew_val, (10, 1))
            ts_blocks.extend([slope_block, skew_block])
        
        ts_input = np.concatenate(ts_blocks, axis=1)
        
        t_ts = torch.from_numpy(ts_input).unsqueeze(0).float()
        
        pred_ts = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            for m in self.tsmixer_models:
                pred_ts += m(t_ts).squeeze().numpy()
        pred_ts /= len(self.tsmixer_models)
        pred_ts = np.clip(pred_ts, self.clip_min_ts.min(), self.clip_max_ts.max())
        
        # --- Blend (TSMixer + MLP v19 + MLP v21) ---
        final_pred = 0.60 * pred_ts + 0.20 * pred_v19 + 0.20 * pred_v21
        
        return np.nan_to_num(final_pred, nan=0.0, posinf=0.0, neginf=0.0)

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import ScorerStepByStep
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'datasets', 'train.parquet')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.path.dirname(base_dir), 'datasets', 'train.parquet')
    print("Scoring Grand Ensemble v30 locally...")
    scorer = ScorerStepByStep(dataset_path)
    res = scorer.score(PredictionModel())
    print(f"Mean R2: {res['mean_r2']:.6f}")
