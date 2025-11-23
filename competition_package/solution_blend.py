import os
from typing import Optional

import numpy as np
import torch
from torch import nn

from utils import DataPoint, ScorerStepByStep

BLEND_ALPHA = float(os.environ.get("BLEND_ALPHA", "0.6"))  # weight on level model


class LagMLP(nn.Module):
    """Same architecture as v19 training."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class FeatureBuilder:
    """Streaming feature builder (v19 feature set) that returns the raw feature vector (pre-normalization)."""

    def __init__(self, clip_min: np.ndarray, clip_max: np.ndarray, n_lags: int) -> None:
        self.clip_min = clip_min.astype(np.float32)
        self.clip_max = clip_max.astype(np.float32)
        self.n_lags = n_lags
        self.current_seq_ix: Optional[int] = None
        self.state_history: list[np.ndarray] = []

    def reset(self, seq_ix: int) -> None:
        self.current_seq_ix = seq_ix
        self.state_history = []

    def _build_raw_features(self, data_point: DataPoint) -> Optional[np.ndarray]:
        if len(self.state_history) < self.n_lags:
            return None

        raw_slice = np.stack(self.state_history[-self.n_lags :], axis=0).astype(np.float32)
        lag_slice = np.clip(raw_slice, self.clip_min, self.clip_max)

        lag_flat = lag_slice.reshape(-1)

        last = lag_slice[-1]
        delta_slice = lag_slice - last
        delta_flat = delta_slice.reshape(-1)

        mean_last = lag_slice.mean(axis=0).astype(np.float32)
        std_last = lag_slice.std(axis=0).astype(np.float32)

        x_lag = lag_slice[:-1, :]
        y_lag = lag_slice[1:, :]

        x_mean = x_lag.mean(axis=0)
        y_mean = y_lag.mean(axis=0)

        x_center = x_lag - x_mean
        y_center = y_lag - y_mean

        num = (x_center * y_center).mean(axis=0)
        denom = np.sqrt((x_center**2).mean(axis=0) * (y_center**2).mean(axis=0)) + 1e-8
        ac_lag1 = (num / denom).astype(np.float32)

        if lag_slice.shape[0] > 2:
            x_lag2 = lag_slice[:-2, :]
            y_lag2 = lag_slice[2:, :]
            x2_mean = x_lag2.mean(axis=0)
            y2_mean = y_lag2.mean(axis=0)
            x2_center = x_lag2 - x2_mean
            y2_center = y_lag2 - y2_mean
            num2 = (x2_center * y2_center).mean(axis=0)
            denom2 = np.sqrt((x2_center**2).mean(axis=0) * (y2_center**2).mean(axis=0)) + 1e-8
            ac_lag2 = (num2 / denom2).astype(np.float32)
        else:
            ac_lag2 = np.zeros_like(ac_lag1, dtype=np.float32)

        if lag_slice.shape[0] > 3:
            x_lag3 = lag_slice[:-3, :]
            y_lag3 = lag_slice[3:, :]
            x3_mean = x_lag3.mean(axis=0)
            y3_mean = y_lag3.mean(axis=0)
            x3_center = x_lag3 - x3_mean
            y3_center = y_lag3 - y3_mean
            num3 = (x3_center * y3_center).mean(axis=0)
            denom3 = np.sqrt((x3_center**2).mean(axis=0) * (y3_center**2).mean(axis=0)) + 1e-8
            ac_lag3 = (num3 / denom3).astype(np.float32)
        else:
            ac_lag3 = np.zeros_like(ac_lag1, dtype=np.float32)

        acf_sum_1_3 = (np.abs(ac_lag1) + np.abs(ac_lag2) + np.abs(ac_lag3)).astype(np.float32)

        above = lag_slice > mean_last[None, :]
        frac_above = above.mean(axis=0).astype(np.float32)

        q25 = np.percentile(lag_slice, 25, axis=0).astype(np.float32)
        median = np.percentile(lag_slice, 50, axis=0).astype(np.float32)
        q75 = np.percentile(lag_slice, 75, axis=0).astype(np.float32)
        iqr = (q75 - q25).astype(np.float32)

        denom_std = std_last + 1e-8
        standardized = (lag_slice - mean_last[None, :]) / denom_std[None, :]
        skewness = (standardized**3).mean(axis=0).astype(np.float32)
        kurtosis = ((standardized**4).mean(axis=0) - 3.0).astype(np.float32)
        cv = (std_last / (np.abs(mean_last) + 1e-8)).astype(np.float32)

        n_lags = lag_slice.shape[0]
        t = np.arange(n_lags, dtype=np.float32)
        sum_t = float(n_lags * (n_lags - 1) / 2.0)
        sum_t2 = float(n_lags * (n_lags - 1) * (2 * n_lags - 1) / 6.0)
        sum_y = lag_slice.sum(axis=0)
        sum_ty = (t[:, None] * lag_slice).sum(axis=0)
        denom_trend = n_lags * sum_t2 - sum_t * sum_t
        if denom_trend == 0.0:
            trend_slope = np.zeros_like(mean_last, dtype=np.float32)
            trend_r2 = np.zeros_like(mean_last, dtype=np.float32)
        else:
            trend_slope = (n_lags * sum_ty - sum_t * sum_y) / denom_trend
            trend_slope = trend_slope.astype(np.float32)
            intercept = (sum_y - trend_slope * sum_t) / float(n_lags)
            fitted = intercept[None, :] + trend_slope[None, :] * t[:, None]
            residual = lag_slice - fitted
            ss_res = (residual**2).sum(axis=0)
            mean_y = lag_slice.mean(axis=0)
            ss_tot = ((lag_slice - mean_y[None, :]) ** 2).sum(axis=0)
            trend_r2 = (1.0 - ss_res / (ss_tot + 1e-8)).astype(np.float32)

        mid = n_lags // 2
        if mid == 0 or mid == n_lags:
            curvature = np.zeros_like(mean_last, dtype=np.float32)
        else:
            first_span = float(mid)
            second_span = float(n_lags - mid)
            slope_first = (lag_slice[mid, :] - lag_slice[0, :]) / first_span
            slope_second = (lag_slice[-1, :] - lag_slice[mid, :]) / second_span
            curvature = (slope_second - slope_first).astype(np.float32)

        step_feature = np.array([data_point.step_in_seq / 1000.0], dtype=np.float32)

        x = np.concatenate(
            [
                lag_flat, delta_flat, mean_last, std_last,
                ac_lag1, ac_lag2, ac_lag3, acf_sum_1_3, frac_above,
                q25, median, q75, iqr, skewness, kurtosis, cv,
                trend_slope, trend_r2, curvature,
                step_feature,
            ],
            axis=0,
        )
        return x


class PredictionModel:
    """
    Blended level + residual MLPs:
    - Level model: standard v19 (predicts next level).
    - Residual model: trained on deltas; reconstructed as current_state + delta.
    - Final prediction: alpha * level + (1 - alpha) * (state + delta_pred).
    """

    def __init__(self, alpha: float = BLEND_ALPHA) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, "models")

        # Load level normalization
        norm_level_path = os.path.join(weights_dir, "lag_mlp_normalization.npz")
        if not os.path.exists(norm_level_path):
            raise FileNotFoundError("Missing lag_mlp_normalization.npz for level model.")
        norm_level = np.load(norm_level_path, allow_pickle=True)
        self.level_mean = norm_level["x_mean"].astype(np.float32)
        self.level_std = norm_level["x_std"].astype(np.float32)
        self.clip_min = norm_level["clip_min"].astype(np.float32)
        self.clip_max = norm_level["clip_max"].astype(np.float32)
        self.n_lags: int = int(norm_level["n_lags"])

        # Residual normalization (fallback to level stats if not found)
        norm_res_path = os.path.join(weights_dir, "lag_mlp_residual_normalization.npz")
        if os.path.exists(norm_res_path):
            norm_res = np.load(norm_res_path, allow_pickle=True)
            self.res_mean = norm_res["x_mean"].astype(np.float32)
            self.res_std = norm_res["x_std"].astype(np.float32)
        else:
            self.res_mean = self.level_mean
            self.res_std = self.level_std

        # Load level ensemble
        self.level_models: list[LagMLP] = []
        level_paths = sorted(
            [
                os.path.join(weights_dir, f)
                for f in os.listdir(weights_dir)
                if f.startswith("lag_mlp_fold") and f.endswith(".pth")
            ]
        )
        if not level_paths:
            raise FileNotFoundError("No level model weights found (lag_mlp_fold*.pth).")
        first_ckpt = torch.load(level_paths[0], map_location="cpu")
        input_dim = int(first_ckpt["input_dim"])
        hidden_dim = int(first_ckpt["hidden_dim"])
        output_dim = int(first_ckpt["output_dim"])
        for path in level_paths:
            ckpt = torch.load(path, map_location="cpu")
            m = LagMLP(input_dim, hidden_dim, output_dim)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.level_models.append(m)

        # Load residual ensemble if available
        self.residual_models: list[LagMLP] = []
        res_paths = sorted(
            [
                os.path.join(weights_dir, f)
                for f in os.listdir(weights_dir)
                if f.startswith("lag_mlp_residual_fold") and f.endswith(".pth")
            ]
        )
        for path in res_paths:
            ckpt = torch.load(path, map_location="cpu")
            m = LagMLP(input_dim, hidden_dim, output_dim)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.residual_models.append(m)

        torch.set_num_threads(1)
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.builder = FeatureBuilder(self.clip_min, self.clip_max, self.n_lags)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.builder.current_seq_ix != data_point.seq_ix:
            self.builder.reset(data_point.seq_ix)

        self.builder.state_history.append(data_point.state.astype(np.float32).copy())
        if len(self.builder.state_history) > self.n_lags:
            self.builder.state_history = self.builder.state_history[-self.n_lags :]

        if not data_point.need_prediction:
            return None

        x_raw = self.builder._build_raw_features(data_point)
        if x_raw is None:
            return data_point.state.astype(np.float32)

        if x_raw.shape[0] != self.level_mean.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch ({x_raw.shape[0]} vs expected {self.level_mean.shape[0]}). "
                "Ensure weights and normalization match the feature set."
            )

        with torch.no_grad():
            x_level = torch.from_numpy((x_raw - self.level_mean) / self.level_std).unsqueeze(0)
            level_preds = [m(x_level).squeeze(0).cpu().numpy() for m in self.level_models]
            level_pred = np.mean(level_preds, axis=0)

            if self.residual_models:
                x_res = torch.from_numpy((x_raw - self.res_mean) / self.res_std).unsqueeze(0)
                res_preds = [m(x_res).squeeze(0).cpu().numpy() for m in self.residual_models]
                delta_pred = np.mean(res_preds, axis=0)
                blended = self.alpha * level_pred + (1.0 - self.alpha) * (data_point.state + delta_pred)
                return blended.astype(np.float32)
            else:
                return level_pred.astype(np.float32)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")

    print(f"Loading dataset and evaluating blended PredictionModel (alpha={BLEND_ALPHA}) on train.parquet...")
    scorer = ScorerStepByStep(dataset_path)
    model = PredictionModel()
    results = scorer.score(model)

    print("\nResults on train.parquet:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")
