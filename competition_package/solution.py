import os
from typing import Optional

import numpy as np
import torch
from torch import nn

from utils import DataPoint, ScorerStepByStep


class LagMLP(nn.Module):
    """Small MLP that mirrors the architecture used in train_model.py."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Funnel-style architecture: input_dim -> 2 * hidden_dim -> hidden_dim -> output_dim
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3), # Matches tuned dropout
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3), # Matches tuned dropout
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class PredictionModel:
    """
    Lag-based neural model (v13 Kinematics & Volatility).

    - Maintains a rolling window of the last `n_lags` states.
    - Applies WINSORIZATION (Clipping) to the lag window.
    - Builds streaming-safe features (including Accel/VolExp/Roughness).
    - Ensembles 5 models trained via K-Fold CV.
    """

    def __init__(self) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, "models")
        norm_path = os.path.join(weights_dir, "lag_mlp_normalization.npz")

        if not os.path.exists(norm_path):
            raise FileNotFoundError(
                "Normalization file not found. "
                "Run train_model.py to generate models/lag_mlp_normalization.npz before scoring."
            )

        # Load normalization and winsorization parameters
        norm = np.load(norm_path, allow_pickle=True)
        self.x_mean = norm["x_mean"].astype(np.float32)
        self.x_std = norm["x_std"].astype(np.float32)
        self.clip_min = norm["clip_min"].astype(np.float32)
        self.clip_max = norm["clip_max"].astype(np.float32)
        self.n_lags: int = int(norm["n_lags"])
        
        # Try to load the CV ensemble (lag_mlp_fold*.pth)
        self.models: list[LagMLP] = []
        
        if os.path.isdir(weights_dir):
            fold_paths = sorted(
                [
                    os.path.join(weights_dir, fname)
                    for fname in os.listdir(weights_dir)
                    if fname.startswith("lag_mlp_fold") and fname.endswith(".pth")
                ]
            )
        else:
            fold_paths = []

        if not fold_paths:
             # Fallback
            seed_paths = sorted([
                os.path.join(weights_dir, f) for f in os.listdir(weights_dir)
                if f.startswith("lag_mlp_seed") and f.endswith(".pth")
            ])
            fold_paths = seed_paths if seed_paths else [os.path.join(weights_dir, "lag_mlp.pth")]

        if not os.path.exists(fold_paths[0]):
             raise FileNotFoundError("No trained model files found in models/ directory.")

        # Read architecture from the first model
        first_ckpt = torch.load(fold_paths[0], map_location="cpu")
        input_dim = int(first_ckpt["input_dim"])
        hidden_dim = int(first_ckpt["hidden_dim"])
        output_dim = int(first_ckpt["output_dim"])

        for path in fold_paths:
            ckpt = torch.load(path, map_location="cpu")
            model = LagMLP(input_dim, hidden_dim, output_dim)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.models.append(model)

        torch.set_num_threads(1)
        self.current_seq_ix: Optional[int] = None
        self.state_history: list[np.ndarray] = []

    def _reset_sequence(self, seq_ix: int) -> None:
        self.current_seq_ix = seq_ix
        self.state_history = []

    def _build_features(self, data_point: DataPoint) -> Optional[np.ndarray]:
        """
        Build normalized feature vector for the current point.
        APPLIES WINSORIZATION (CLIPPING) to the raw lag window.
        """
        if len(self.state_history) < self.n_lags:
            return None

        # 1. Extract Raw Window
        raw_slice = np.stack(self.state_history[-self.n_lags :], axis=0).astype(np.float32)

        # 2. Apply Winsorization (Clipping)
        lag_slice = np.clip(raw_slice, self.clip_min, self.clip_max)

        # 3. Build Features
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

        # lag-2 autocorrelation
        if lag_slice.shape[0] > 2:
            x_lag2 = lag_slice[:-2, :]
            y_lag2 = lag_slice[2:, :]
            x2_mean = x_lag2.mean(axis=0)
            y2_mean = y_lag2.mean(axis=0)
            x2_center = x_lag2 - x2_mean
            y2_center = y_lag2 - y2_mean
            num2 = (x2_center * y2_center).mean(axis=0)
            denom2 = (np.sqrt((x2_center**2).mean(axis=0) * (y2_center**2).mean(axis=0)) + 1e-8)
            ac_lag2 = (num2 / denom2).astype(np.float32)
        else:
            ac_lag2 = np.zeros_like(ac_lag1, dtype=np.float32)

        # lag-3 autocorrelation
        if lag_slice.shape[0] > 3:
            x_lag3 = lag_slice[:-3, :]
            y_lag3 = lag_slice[3:, :]
            x3_mean = x_lag3.mean(axis=0)
            y3_mean = y_lag3.mean(axis=0)
            x3_center = x_lag3 - x3_mean
            y3_center = y_lag3 - y3_mean
            num3 = (x3_center * y3_center).mean(axis=0)
            denom3 = (np.sqrt((x3_center**2).mean(axis=0) * (y3_center**2).mean(axis=0)) + 1e-8)
            ac_lag3 = (num3 / denom3).astype(np.float32)
        else:
            ac_lag3 = np.zeros_like(ac_lag1, dtype=np.float32)

        acf_sum_1_3 = (np.abs(ac_lag1) + np.abs(ac_lag2) + np.abs(ac_lag3)).astype(np.float32)

        above = lag_slice > mean_last[None, :]
        frac_above = above.mean(axis=0).astype(np.float32)

        # Robust stats
        q25 = np.percentile(lag_slice, 25, axis=0).astype(np.float32)
        median = np.percentile(lag_slice, 50, axis=0).astype(np.float32)
        q75 = np.percentile(lag_slice, 75, axis=0).astype(np.float32)
        iqr = (q75 - q25).astype(np.float32)

        denom_std = std_last + 1e-8
        standardized = (lag_slice - mean_last[None, :]) / denom_std[None, :]
        skewness = (standardized**3).mean(axis=0).astype(np.float32)
        kurtosis = ((standardized**4).mean(axis=0) - 3.0).astype(np.float32)
        cv = (std_last / (np.abs(mean_last) + 1e-8)).astype(np.float32)

        # Trend
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
            
        # v13 New Features
        # Volatility Expansion
        half_idx = n_lags // 2
        std_recent = lag_slice[half_idx:].std(axis=0)
        vol_exp = (std_recent / (std_last + 1e-8)).astype(np.float32)
        
        # Roughness
        diffs = np.diff(lag_slice, axis=0)
        path_len = np.sum(np.abs(diffs), axis=0)
        displacement = np.abs(lag_slice[-1] - lag_slice[0])
        roughness = (path_len / (displacement + 1e-8)).astype(np.float32)
        
        # Acceleration
        vel = np.diff(lag_slice, axis=0)
        acc = np.diff(vel, axis=0)
        accel_mean = acc.mean(axis=0).astype(np.float32)

        step_feature = np.array([data_point.step_in_seq / 1000.0], dtype=np.float32)

        # Concatenate
        x = np.concatenate(
            [
                lag_flat, delta_flat, mean_last, std_last,
                ac_lag1, ac_lag2, ac_lag3, acf_sum_1_3, frac_above,
                q25, median, q75, iqr, skewness, kurtosis, cv,
                trend_slope, trend_r2, curvature,
                vol_exp, roughness, accel_mean,
                step_feature,
            ],
            axis=0,
        )

        # Normalize
        x_norm = (x - self.x_mean) / self.x_std
        return x_norm

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Reset recurrent state when a new sequence starts
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_sequence(data_point.seq_ix)

        # Always store current state; keep only the latest n_lags entries
        self.state_history.append(data_point.state.astype(np.float32).copy())
        if len(self.state_history) > self.n_lags:
            self.state_history = self.state_history[-self.n_lags :]

        # If prediction is not required for this step, return None
        if not data_point.need_prediction:
            return None

        # Try to build lag features; if not enough history, fall back to current state
        x_norm = self._build_features(data_point)
        if x_norm is None:
            return data_point.state.astype(np.float32)

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0)  # (1, input_dim)
            # Ensemble Average
            preds_list = [model(x_tensor).squeeze(0).cpu().numpy() for model in self.models]
            preds = np.mean(preds_list, axis=0)

        return preds


if __name__ == "__main__":
    # Optional local test: evaluate this model on the training data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")

    print("Loading dataset and evaluating PredictionModel on train.parquet...")
    scorer = ScorerStepByStep(dataset_path)
    model = PredictionModel()

    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    results = scorer.score(model)

    print("\nResults on train.parquet:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")