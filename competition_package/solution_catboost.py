import os
from typing import Optional

import numpy as np
from catboost import CatBoostRegressor

from utils import DataPoint, ScorerStepByStep

# Number of lags used in the CatBoost training.
# This must match the value used in train_catboost_experiment.py /
# build_supervised_dataset when the model was trained.
N_LAGS_DEFAULT = 10


def _build_raw_features_from_history(
    state_history: list[np.ndarray],
    n_lags: int,
    step_in_seq: int,
) -> Optional[np.ndarray]:
    """
    Build the raw (unnormalized) v5 feature vector from the recent
    state history and current step index.

    This mirrors the feature construction used in:
      - train_model.build_supervised_dataset (without any normalization)
      - solution.PredictionModel._build_features before applying x_mean/x_std.
    """
    if len(state_history) < n_lags:
        return None

    # Use last n_lags states
    history_window = state_history[-n_lags:]
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

    step_feature = np.array([step_in_seq / 1000.0], dtype=np.float32)

    x = np.concatenate(
        [lag_flat, delta_flat, mean_last, std_last, ac_lag1, frac_above, step_feature],
        axis=0,
    )
    return x


class PredictionModel:
    """
    Streaming CatBoost model using the v5 feature set:

      [lag_flat, delta_flat, mean_last10, std_last10,
       ac_lag1, frac_above_mean, step_in_seq/1000]

    The CatBoostRegressor is trained offline via train_catboost_experiment.py
    on the same features built by train_model.build_supervised_dataset.

    NOTE: This file is intended as an alternative submission entry.
    For leaderboard submission, rename/copy this file to solution.py
    so that the competition runner picks up this CatBoost-based
    PredictionModel instead of the MLP version.
    """

    def __init__(self) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, "models")
        model_path = os.path.join(weights_dir, "catboost_lag_delta_multiRMSE.cbm")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"CatBoost model file not found at {model_path}. "
                "Run train_catboost_experiment.py to train and save the model "
                "before scoring or submitting this solution."
            )

        self.model = CatBoostRegressor()
        self.model.load_model(model_path)

        # Use the same lag window length as in CatBoost training.
        self.n_lags: int = int(N_LAGS_DEFAULT)

        self.current_seq_ix: Optional[int] = None
        self.state_history: list[np.ndarray] = []

    def _reset_sequence(self, seq_ix: int) -> None:
        self.current_seq_ix = seq_ix
        self.state_history = []

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Reset history when a new sequence starts
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_sequence(data_point.seq_ix)

        # Update per-sequence history and keep only the latest n_lags entries
        self.state_history.append(data_point.state.astype(np.float32).copy())
        if len(self.state_history) > self.n_lags:
            self.state_history = self.state_history[-self.n_lags :]

        # If no prediction is required at this step, just update state and return None.
        if not data_point.need_prediction:
            return None

        # Build raw v5 feature vector from history; if not enough history yet,
        # fall back to returning the current state as a simple baseline.
        x = _build_raw_features_from_history(
            self.state_history, self.n_lags, data_point.step_in_seq
        )
        if x is None:
            return data_point.state.astype(np.float32)

        # CatBoost expects 2D input: (n_samples, n_features).
        preds = self.model.predict(x[None, :])
        # Output is (1, 32); squeeze to (32,) and cast to float32 for consistency.
        return np.asarray(preds[0], dtype=np.float32)


if __name__ == "__main__":
    # Local test: evaluate this CatBoost-based model on train.parquet
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")

    print("Loading dataset and evaluating CatBoost PredictionModel on train.parquet...")
    scorer = ScorerStepByStep(dataset_path)
    model = PredictionModel()

    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    results = scorer.score(model)

    print("\nResults on train.parquet (CatBoost v5 features):")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")
