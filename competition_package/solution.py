import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

# --- TSMixer Classes (Must be included for loading) ---
class TimeBatchNorm2d(nn.BatchNorm1d):
    def __init__(self, normalized_shape: tuple):
        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor, but got {x.ndim}D tensor instead.")
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

# --- Feature Extractor Logic ---

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

# --- Main Prediction Model ---

# Try to import DataPoint from local or root
try:
    from utils import DataPoint
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import DataPoint

class PredictionModel:
    def __init__(self):
        self.device = torch.device("cpu")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Adaptive models directory
        models_dir = os.path.join(base_dir, "models")
        if not os.path.exists(models_dir):
            # Fallback for local testing
            models_dir = os.path.join(base_dir, "..", "models")
            
        # --- 1. Load MLP v19 (Level) ---
        self.mlp_v19_models = []
        norm_v19 = np.load(os.path.join(models_dir, "lag_mlp_normalization.npz"))
        self.mean_v19 = norm_v19["x_mean"].astype(np.float32)
        self.std_v19 = norm_v19["x_std"].astype(np.float32)
        self.clip_min_v19 = norm_v19["clip_min"].astype(np.float32)
        self.clip_max_v19 = norm_v19["clip_max"].astype(np.float32)
        
        for f in sorted([x for x in os.listdir(models_dir) if x.startswith("lag_mlp_fold") and x.endswith(".pth")]):
            ckpt = torch.load(os.path.join(models_dir, f), map_location=self.device)
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
        
        for f in sorted([x for x in os.listdir(models_dir) if x.startswith("lag_mlp_v21_fold") and x.endswith(".pth")]):
            ckpt = torch.load(os.path.join(models_dir, f), map_location=self.device)
            m = LagMLP_v21(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["output_dim"])
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.mlp_v21_models.append(m)
            
        # --- 3. Load TSMixer v2 (Delta) ---
        self.tsmixer_models = []
        norm_ts = np.load(os.path.join(models_dir, "tsmixer_v2_delta_normalization.npz"))
        self.mean_ts = norm_ts["x_mean"].astype(np.float32)
        self.std_ts = norm_ts["x_std"].astype(np.float32)
        self.clip_min_ts = norm_ts["clip_min"].astype(np.float32)
        self.clip_max_ts = norm_ts["clip_max"].astype(np.float32)
        self.ts_use_deltas = True
        
        for f in sorted([x for x in os.listdir(models_dir) if x.startswith("tsmixer_v2_delta_fold") and x.endswith(".pth")]):
            ckpt = torch.load(os.path.join(models_dir, f), map_location=self.device)
            m = TSMixer(
                sequence_length=10,
                prediction_length=1,
                input_channels=64,
                output_channels=32,
                num_blocks=2,
                ff_dim=256,
                dropout_rate=0.17,
                activation_fn="relu",
                normalize_before=True,
                norm_type="batch"
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
            # Impute missing history with zeros
            return np.nan_to_num(data_point.state, nan=0.0)
            
        # Robust History Handling using Pandas
        recent_hist = self.history[-20:] 
        df_hist = pd.DataFrame(np.array(recent_hist, dtype=np.float32))
        df_hist = df_hist.ffill().bfill()
        df_hist = df_hist.fillna(0.0)
        
        raw_window = df_hist.values[-10:] # (10, 32)
        step_val = np.array([data_point.step_in_seq / 1000.0], dtype=np.float32)
        
        # --- 1. MLP v19 (Level) ---
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
        # Clip prediction to known data range
        pred_v19 = np.clip(pred_v19, self.clip_min_v19.min(), self.clip_max_v19.max())
        
        # --- 2. MLP v21 (Residual) ---
        win_v21 = np.clip(raw_window, self.clip_min_v21, self.clip_max_v21)
        feats_v21 = self._build_mlp_features(win_v21, use_spreads=True)
        feats_v21 = np.concatenate([feats_v21, step_val])
        
        x_v21 = (feats_v21 - self.mean_v21) / self.std_v21
        x_v21 = np.nan_to_num(x_v21, nan=0.0, posinf=0.0, neginf=0.0)
        t_v21 = torch.from_numpy(x_v21).unsqueeze(0).float()
        
        pred_v21_resid = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            for m in self.mlp_v21_models:
                pred_v21_resid += m(t_v21).squeeze().numpy()
        pred_v21_resid /= len(self.mlp_v21_models)
        
        # Add residual to last CLEAN state
        last_state = raw_window[-1]
        pred_v21 = last_state + pred_v21_resid
        pred_v21 = np.clip(pred_v21, self.clip_min_v21.min(), self.clip_max_v21.max())
        
        # --- 3. TSMixer v2 (Delta) ---
        win_ts = np.clip(raw_window, self.clip_min_ts, self.clip_max_ts)
        diffs = np.zeros_like(win_ts)
        diffs[1:] = win_ts[1:] - win_ts[:-1]
        ts_input = np.concatenate([win_ts, diffs], axis=1)
        
        x_ts = (ts_input - self.mean_ts) / self.std_ts
        x_ts = np.nan_to_num(x_ts, nan=0.0, posinf=0.0, neginf=0.0)
        
        t_ts = torch.from_numpy(x_ts).unsqueeze(0).float()
        
        pred_ts = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            for m in self.tsmixer_models:
                pred_ts += m(t_ts).squeeze().numpy()
        pred_ts /= len(self.tsmixer_models)
        pred_ts = np.clip(pred_ts, self.clip_min_ts.min(), self.clip_max_ts.max())
        
        # --- Triplet Blend ---
        final_pred = 0.50 * pred_v19 + 0.35 * pred_v21 + 0.15 * pred_ts
        
        return np.nan_to_num(final_pred, nan=0.0, posinf=0.0, neginf=0.0)

if __name__ == "__main__":
    # Ensure root in path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import ScorerStepByStep
    
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'train.parquet')
    print(f"Scoring triplet ensemble on {dataset_path}...")
    
    # We will wrap the model to print debug info for the first few steps
    model = PredictionModel()
    
    # Patch predict to print stats for first step
    original_predict = model.predict
    def debug_predict(data_point):
        res = original_predict(data_point)
        if res is not None and data_point.step_in_seq == 100 and data_point.seq_ix == 0:
            print(f"[Debug] Step 100, Seq 0:")
            print(f"  Input State Mean: {data_point.state.mean():.4f}, Std: {data_point.state.std():.4f}")
            print(f"  Prediction Mean: {res.mean():.4f}, Std: {res.std():.4f}, Min: {res.min():.4f}, Max: {res.max():.4f}")
        return res
    
    model.predict = debug_predict
    
    scorer = ScorerStepByStep(dataset_path)
    results = scorer.score(model)
    print(f"Mean R2: {results['mean_r2']:.6f}")