import numpy as np
from .math_utils import (
    compute_frac_above_mean,
    compute_lagk_autocorr,
    compute_robust_window_stats,
    compute_trend_features,
)


class FeatureExtractor:
    """
    Shared feature builder matching the v19 streaming-safe set (lags/deltas/rolling/robust/stats/trend/step).
    Can be used in batch (offline) or streaming mode.
    """

    def __init__(
        self, 
        n_lags: int = 10, 
        clip_min: np.ndarray | None = None, 
        clip_max: np.ndarray | None = None,
        use_spreads: bool = False
    ):
        self.n_lags = n_lags
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.use_spreads = use_spreads
        self.buffer: list[np.ndarray] = []
        self.current_seq = None

    def _clip(self, arr: np.ndarray) -> np.ndarray:
        if self.clip_min is None or self.clip_max is None:
            return arr.astype(np.float32)
        return np.clip(arr, self.clip_min, self.clip_max).astype(np.float32)

    def reset(self):
        self.buffer = []
        self.current_seq = None

    def build_window_features(self, window: np.ndarray, step_in_seq: int) -> np.ndarray:
        """
        Build v19 feature vector for a given window (shape: [n_lags, 32]).
        """
        assert window.shape[0] == self.n_lags, f"Expected window len {self.n_lags}, got {window.shape[0]}"
        lag_slice = self._clip(window)
        lag_flat = lag_slice.reshape(-1)
        last = lag_slice[-1]
        delta_flat = (lag_slice - last).reshape(-1)
        mean_last = lag_slice.mean(axis=0).astype(np.float32)
        std_last = lag_slice.std(axis=0).astype(np.float32)

        ac1 = compute_lagk_autocorr(lag_slice, 1)
        ac2 = compute_lagk_autocorr(lag_slice, 2)
        ac3 = compute_lagk_autocorr(lag_slice, 3)
        acf_sum = np.abs(ac1) + np.abs(ac2) + np.abs(ac3)

        frac = compute_frac_above_mean(lag_slice, mean_last)
        q25, median, q75, iqr, skew, kurt, cv = compute_robust_window_stats(lag_slice, mean_last, std_last)
        slope, r2, curve = compute_trend_features(lag_slice)
        step_val = np.array([step_in_seq / 1000.0], dtype=np.float32)

        feature_list = [
            lag_flat,
            delta_flat,
            mean_last,
            std_last,
            ac1,
            ac2,
            ac3,
            acf_sum,
            frac,
            q25,
            median,
            q75,
            iqr,
            skew,
            kurt,
            cv,
            slope,
            r2,
            curve,
            step_val,
        ]

        if self.use_spreads:
            # Highly correlated pairs identified in EDA: (18, 28) and (1, 28)
            # Compute stationary spreads from the most recent state
            spread_18_28 = last[18] - last[28]
            spread_1_28 = last[1] - last[28]
            spreads = np.array([spread_18_28, spread_1_28], dtype=np.float32)
            feature_list.append(spreads)

        features = np.concatenate(feature_list).astype(np.float32)
        return features

    def stream(self, state: np.ndarray, step_in_seq: int, seq_ix: int | None = None) -> np.ndarray | None:
        """
        Append state to buffer and, if enough history is present, return features; else None.
        """
        if seq_ix is not None and seq_ix != self.current_seq:
            self.reset()
            self.current_seq = seq_ix

        self.buffer.append(state.astype(np.float32))
        if len(self.buffer) < self.n_lags:
            return None
        if len(self.buffer) > self.n_lags:
            self.buffer = self.buffer[-self.n_lags :]

        window = np.stack(self.buffer, axis=0)
        return self.build_window_features(window, step_in_seq)


def feature_dim(n_lags: int = 10, use_spreads: bool = False) -> int:
    """Return the expected feature dimension for v19 layout."""
    base = 32
    dim = (
        n_lags * base  # lags
        + n_lags * base  # deltas
        + base  # mean
        + base  # std
        + base * 4  # ac1, ac2, ac3, acf_sum
        + base  # frac
        + base * 3  # q25, median, q75
        + base  # iqr
        + base  # skew
        + base  # kurt
        + base  # cv
        + base  # slope
        + base  # r2
        + base  # curve
        + 1  # step_val
    )
    if use_spreads:
        dim += 2
    return dim
