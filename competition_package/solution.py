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
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class PredictionModel:
    """
    Lag-based neural model.

    - Maintains a rolling window of the last `n_lags` states per sequence.
    - When a prediction is required and enough history is available, builds
      the same feature vector as in train_model.py:
        [flattened last n_lags states, step_in_seq / 1000.0]
      applies saved normalization and feeds it into a small MLP.
    - For early steps without enough history, falls back to returning the
      current state (simple baseline) to satisfy the interface.
    """

    def __init__(self) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, "models")
        model_path = os.path.join(weights_dir, "lag_mlp.pth")
        norm_path = os.path.join(weights_dir, "lag_mlp_normalization.npz")

        if not (os.path.exists(model_path) and os.path.exists(norm_path)):
            raise FileNotFoundError(
                "Trained model files not found. "
                "Run train_model.py to generate models/lag_mlp.pth "
                "and models/lag_mlp_normalization.npz before scoring."
            )

        # Load normalization parameters
        norm = np.load(norm_path, allow_pickle=True)
        self.x_mean = norm["x_mean"].astype(np.float32)
        self.x_std = norm["x_std"].astype(np.float32)
        self.n_lags: int = int(norm["n_lags"])
        # feature_cols is stored for reference; we don't use names at inference
        self.feature_cols = norm["feature_cols"].tolist()

        checkpoint = torch.load(model_path, map_location="cpu")
        input_dim = int(checkpoint["input_dim"])
        hidden_dim = int(checkpoint["hidden_dim"])
        output_dim = int(checkpoint["output_dim"])

        self.model = LagMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        # Use single-threaded CPU for determinism and resource friendliness
        torch.set_num_threads(1)

        self.current_seq_ix: Optional[int] = None
        self.state_history: list[np.ndarray] = []

    def _reset_sequence(self, seq_ix: int) -> None:
        self.current_seq_ix = seq_ix
        self.state_history = []

    def _build_features(self, data_point: DataPoint) -> Optional[np.ndarray]:
        """
        Build normalized feature vector for the current point, or return None
        if there is not enough history.
        """
        if len(self.state_history) < self.n_lags:
            return None

        # Use last n_lags states
        history_window = self.state_history[-self.n_lags :]
        lag_slice = np.stack(history_window, axis=0).astype(np.float32)  # (n_lags, dim)

        # raw lags: flatten window
        lag_flat = lag_slice.reshape(-1)

        # LastKnown-delta features: subtract last lag (most recent state)
        last = lag_slice[-1]  # (dim,)
        delta_slice = lag_slice - last  # (n_lags, dim)
        delta_flat = delta_slice.reshape(-1)

        # Rolling statistics over the lag window (per feature)
        mean_last = lag_slice.mean(axis=0).astype(np.float32)
        std_last = lag_slice.std(axis=0).astype(np.float32)

        # Streaming-safe analogs inspired by catch22:
        # - lag-1 autocorrelation per feature
        # - fraction of window values above the mean (persistence)
        x_lag = lag_slice[:-1, :]
        y_lag = lag_slice[1:, :]

        x_mean = x_lag.mean(axis=0)
        y_mean = y_lag.mean(axis=0)

        x_center = x_lag - x_mean
        y_center = y_lag - y_mean

        num = (x_center * y_center).mean(axis=0)
        denom = np.sqrt((x_center**2).mean(axis=0) * (y_center**2).mean(axis=0)) + 1e-8
        ac_lag1 = (num / denom).astype(np.float32)

        above = lag_slice > mean_last[None, :]
        frac_above = above.mean(axis=0).astype(np.float32)

        step_feature = np.array([data_point.step_in_seq / 1000.0], dtype=np.float32)

        # v5 feature set for submission: v3 features plus streaming-safe analogs
        # inspired by catch22 (lag-1 autocorr and persistence).
        x = np.concatenate(
            [lag_flat, delta_flat, mean_last, std_last, ac_lag1, frac_above, step_feature],
            axis=0,
        )

        # Normalize with training statistics
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
            preds = self.model(x_tensor).squeeze(0).cpu().numpy()

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
