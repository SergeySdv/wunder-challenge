import os
from typing import Tuple, List, Set, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import nn


N_LAGS_DEFAULT = 10
HIDDEN_SIZE = 256
N_EPOCHS = 20
BATCH_SIZE = 1024
LR = 1e-3
WEIGHTS_DIR = "models"
ENSEMBLE_SEEDS = [42, 43, 44]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Optional: precomputed catch22 features per sequence (offline lab).
# If `datasets/catch22_per_seq.npz` exists, we can load it and build a mapping:
#   seq_ix -> flattened catch22 feature vector.
# For experiments where we only want to keep the most useful catch22
# statistics, we select a subset by name instead of all 22.
_CATCH22_PATH = os.path.join(BASE_DIR, "datasets", "catch22_per_seq.npz")
_SEQ_TO_CATCH22: Optional[Dict[int, np.ndarray]] = None

if os.path.exists(_CATCH22_PATH):
    _catch22_npz = np.load(_CATCH22_PATH, allow_pickle=True)
    _catch22_seq_ids = _catch22_npz["seq_ids"]
    _catch22_values = _catch22_npz["catch22_values"]  # (n_seqs, n_dims, 22)
    _catch22_names = _catch22_npz["catch22_names"].tolist()

    # Subset of catch22 statistics that were most important in
    # CatBoost feature-importance analysis. We keep only these for
    # lab experiments to reduce dimensionality.
    _SELECTED_CATCH22_NAMES = [
        "SP_Summaries_welch_rect_area_5_1",
        "SP_Summaries_welch_rect_centroid",
        "CO_f1ecac",
        "CO_FirstMin_ac",
        "FC_LocalSimple_mean1_tauresrat",
        "FC_LocalSimple_mean3_stderr",
        "CO_trev_1_num",
        "SB_BinaryStats_mean_longstretch1",
        "CO_HistogramAMI_even_2_5",
    ]

    _name_to_idx = {name: i for i, name in enumerate(_catch22_names)}
    _selected_indices = [
        _name_to_idx[name]
        for name in _SELECTED_CATCH22_NAMES
        if name in _name_to_idx
    ]

    if not _selected_indices:
        # Fallback: if names do not match for some reason, keep all stats.
        _selected_indices = list(range(_catch22_values.shape[2]))

    _catch22_subset = _catch22_values[:, :, _selected_indices]  # (n_seqs, n_dims, k)
    _catch22_flat = _catch22_subset.reshape(_catch22_subset.shape[0], -1).astype(
        np.float32
    )
    _SEQ_TO_CATCH22 = {
        int(seq_ix): _catch22_flat[i]
        for i, seq_ix in enumerate(_catch22_seq_ids)
    }


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load and sort the competition dataset."""
    df = pd.read_parquet(dataset_path)
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    return df


def _compute_lag1_autocorr(lag_slice: np.ndarray) -> np.ndarray:
    """
    Compute a simple lag-1 autocorrelation estimate for each feature
    over the lag window.

    Parameters
    ----------
    lag_slice : np.ndarray
        Array of shape (n_lags, dim) with recent states.

    Returns
    -------
    ac : np.ndarray
        Array of shape (dim,) with lag-1 autocorrelation estimates
        in [-1, 1]. If variance is (near) zero for a feature, the
        corresponding value is set close to 0.
    """
    # Use pairs (x_t, x_{t-1}) for t = 1..n_lags-1
    x = lag_slice[:-1, :]
    y = lag_slice[1:, :]

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)

    x_center = x - x_mean
    y_center = y - y_mean

    num = (x_center * y_center).mean(axis=0)
    denom = np.sqrt((x_center**2).mean(axis=0) * (y_center**2).mean(axis=0)) + 1e-8

    ac = num / denom
    return ac.astype(np.float32)


def _compute_lagk_autocorr(lag_slice: np.ndarray, lag: int) -> np.ndarray:
    """
    Compute a simple lag-k autocorrelation estimate for each feature
    over the lag window.

    Parameters
    ----------
    lag_slice : np.ndarray
        Array of shape (n_lags, dim) with recent states.
    lag : int
        Positive lag (>= 1) at which to estimate autocorrelation.

    Returns
    -------
    ac : np.ndarray
        Array of shape (dim,) with lag-k autocorrelation estimates
        in [-1, 1].
    """
    if lag <= 0 or lag >= lag_slice.shape[0]:
        # If the requested lag is not valid for the window length,
        # return zeros as a neutral value.
        return np.zeros(lag_slice.shape[1], dtype=np.float32)

    x = lag_slice[:-lag, :]
    y = lag_slice[lag:, :]

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)

    x_center = x - x_mean
    y_center = y - y_mean

    num = (x_center * y_center).mean(axis=0)
    denom = np.sqrt((x_center**2).mean(axis=0) * (y_center**2).mean(axis=0)) + 1e-8

    ac = num / denom
    return ac.astype(np.float32)


def _compute_frac_above_mean(
    lag_slice: np.ndarray, mean_last: np.ndarray
) -> np.ndarray:
    """
    Compute, for each feature, the fraction of values in the lag window
    that are above the window mean.

    This acts as a simple persistence / imbalance statistic (inspired
    by SB_BinaryStats_mean_longstretch1).
    """
    above = lag_slice > mean_last[None, :]
    frac = above.mean(axis=0)
    return frac.astype(np.float32)


def _compute_robust_window_stats(
    lag_slice: np.ndarray, mean_last: np.ndarray, std_last: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute robust distributional statistics over the lag window
    for each feature: quantiles, IQR, skewness, kurtosis, and
    coefficient of variation.
    """
    q25 = np.percentile(lag_slice, 25, axis=0).astype(np.float32)
    median = np.percentile(lag_slice, 50, axis=0).astype(np.float32)
    q75 = np.percentile(lag_slice, 75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)

    # Avoid division by zero when standard deviation is very small.
    denom = std_last + 1e-8
    standardized = (lag_slice - mean_last[None, :]) / denom[None, :]

    skewness = standardized**3
    skewness = skewness.mean(axis=0).astype(np.float32)

    kurtosis = standardized**4
    kurtosis = kurtosis.mean(axis=0).astype(np.float32) - 3.0
    kurtosis = kurtosis.astype(np.float32)

    cv = std_last / (np.abs(mean_last) + 1e-8)
    cv = cv.astype(np.float32)

    return q25, median, q75, iqr, skewness, kurtosis, cv


def _compute_trend_features(lag_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute simple per-feature trend descriptors over the lag window:
      - slope of a least-squares line vs. time index
      - R^2 of that linear fit
      - a crude curvature indicator (difference between late-half and early-half slopes)
    """
    n_lags, dim = lag_slice.shape
    t = np.arange(n_lags, dtype=np.float32)

    # Precompute sums for the shared time index.
    sum_t = float(n_lags * (n_lags - 1) / 2.0)
    sum_t2 = float(n_lags * (n_lags - 1) * (2 * n_lags - 1) / 6.0)

    sum_y = lag_slice.sum(axis=0)  # (dim,)
    sum_ty = (t[:, None] * lag_slice).sum(axis=0)  # (dim,)

    denom = n_lags * sum_t2 - sum_t * sum_t
    if denom == 0.0:
        slope = np.zeros(dim, dtype=np.float32)
        r2 = np.zeros(dim, dtype=np.float32)
    else:
        slope = (n_lags * sum_ty - sum_t * sum_y) / denom
        slope = slope.astype(np.float32)

        intercept = (sum_y - slope * sum_t) / float(n_lags)
        intercept = intercept.astype(np.float32)

        fitted = intercept[None, :] + slope[None, :] * t[:, None]
        residual = lag_slice - fitted
        ss_res = (residual**2).sum(axis=0)
        mean_y = lag_slice.mean(axis=0)
        ss_tot = ((lag_slice - mean_y[None, :]) ** 2).sum(axis=0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        r2 = r2.astype(np.float32)

    # Curvature: difference between late-half and early-half slopes.
    mid = n_lags // 2
    if mid == 0 or mid == n_lags:
        curvature = np.zeros(dim, dtype=np.float32)
    else:
        first_span = float(mid)
        second_span = float(n_lags - mid)
        slope_first = (lag_slice[mid, :] - lag_slice[0, :]) / first_span
        slope_second = (lag_slice[-1, :] - lag_slice[mid, :]) / second_span
        curvature = (slope_second - slope_first).astype(np.float32)

    return slope, r2, curvature


def split_by_seq(
    df: pd.DataFrame, train_frac: float = 0.8, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int], Set[int]]:
    """
    Split the dataframe into train/validation by whole sequences (seq_ix).

    This avoids any leakage of information across sequences.
    """
    seq_ids = df["seq_ix"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(seq_ids)

    n_train = int(len(seq_ids) * train_frac)
    train_ids = set(seq_ids[:n_train])
    val_ids = set(seq_ids[n_train:])

    df_train = df[df["seq_ix"].isin(train_ids)].copy()
    df_val = df[df["seq_ix"].isin(val_ids)].copy()

    df_train = df_train.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    df_val = df_val.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)

    return df_train, df_val, train_ids, val_ids


def build_supervised_dataset(
    df: pd.DataFrame,
    n_lags: int = N_LAGS_DEFAULT,
    add_step_feature: bool = True,
    use_catch22: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Build a supervised learning dataset from the raw table.

    For each sequence and each step t where:
      - need_prediction == 1
      - there are at least `n_lags` past observations
      - there is a next step within the same sequence

    we create:
      - X_t: concatenation of the last `n_lags` state vectors up to step t,
             optional lag-based stats and optional precomputed per-sequence features.
      - y_t: the next state vector at step t+1.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples, dim)
    meta : pd.DataFrame with columns:
        - seq_ix
        - current_step
        - target_step
        - index_in_seq (position in the per-sequence array)
    feature_cols : list of str
        Names of the state feature columns (0..31).
    """
    # All columns except the first three are state features
    feature_cols = [
        c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")
    ]
    dim = len(feature_cols)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    meta_rows: List[dict] = []

    for seq_ix, df_seq in df.groupby("seq_ix"):
        df_seq = df_seq.sort_values("step_in_seq")
        states = df_seq[feature_cols].values  # (T, dim)
        steps = df_seq["step_in_seq"].values
        need_pred = df_seq["need_prediction"].values

        # Optional catch22 vector for this sequence (same for all samples in seq).
        # Only used when explicitly requested via use_catch22=True.
        catch22_vec: Optional[np.ndarray] = None
        if use_catch22 and _SEQ_TO_CATCH22 is not None:
            catch22_vec = _SEQ_TO_CATCH22.get(int(seq_ix))

        T = len(df_seq)
        # iterate over indices 0..T-2, since we always use next step as target
        for idx in range(T - 1):
            if not need_pred[idx]:
                continue

            current_step = int(steps[idx])
            # require enough history for lags
            if idx < n_lags - 1:
                continue

            # build lag window: indices [idx - n_lags + 1 .. idx]
            lag_slice = states[idx - n_lags + 1 : idx + 1]  # (n_lags, dim)
            lag_slice = lag_slice.astype(np.float32)

            # raw lags: flatten window
            lag_flat = lag_slice.reshape(-1)  # (n_lags * dim,)

            # LastKnown-delta features: subtract last lag (most recent state)
            last = lag_slice[-1]  # (dim,)
            delta_slice = lag_slice - last  # (n_lags, dim)
            delta_flat = delta_slice.reshape(-1)  # (n_lags * dim,)

            # Rolling statistics over the lag window (per feature)
            mean_last = lag_slice.mean(axis=0).astype(np.float32)  # (dim,)
            std_last = lag_slice.std(axis=0).astype(np.float32)  # (dim,)

            # Streaming-safe analogs inspired by catch22 (v5 + v6):
            # - lag-1 autocorrelation estimate per feature (v5)
            # - additional short-window autocorr at lags 2 and 3 (v6)
            # - simple aggregate acf sum over lags 1..3 (v6)
            # - fraction of window values above the mean (persistence, v5)
            # - robust rolling stats and local trend (v6)
            ac_lag1 = _compute_lag1_autocorr(lag_slice)  # (dim,)
            frac_above = _compute_frac_above_mean(lag_slice, mean_last)  # (dim,)

            ac_lag2 = _compute_lagk_autocorr(lag_slice, lag=2)  # (dim,)
            ac_lag3 = _compute_lagk_autocorr(lag_slice, lag=3)  # (dim,)
            acf_sum_1_3 = (np.abs(ac_lag1) + np.abs(ac_lag2) + np.abs(ac_lag3)).astype(
                np.float32
            )  # (dim,)

            (
                q25,
                median,
                q75,
                iqr,
                skewness,
                kurtosis,
                cv,
            ) = _compute_robust_window_stats(lag_slice, mean_last, std_last)

            trend_slope, trend_r2, curvature = _compute_trend_features(lag_slice)

            features = [
                lag_flat,
                delta_flat,
                mean_last,
                std_last,
                ac_lag1,
                ac_lag2,
                ac_lag3,
                acf_sum_1_3,
                frac_above,
                q25,
                median,
                q75,
                iqr,
                skewness,
                kurtosis,
                cv,
                trend_slope,
                trend_r2,
                curvature,
            ]
            if add_step_feature:
                # step position as a simple normalized scalar
                features.append(np.array([current_step / 1000.0], dtype=np.float32))

            # Optionally append precomputed per-sequence catch22 features
            if catch22_vec is not None:
                features.append(catch22_vec.astype(np.float32))

            X_list.append(np.concatenate(features, axis=0))

            # target is the next state in the same sequence
            target_state = states[idx + 1]
            y_list.append(target_state.astype(np.float32))

            meta_rows.append(
                {
                    "seq_ix": int(seq_ix),
                    "current_step": current_step,
                    "target_step": int(steps[idx + 1]),
                    "index_in_seq": int(idx),
                }
            )

    if not X_list:
        raise ValueError("No supervised samples were generated; check n_lags and data.")

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    meta = pd.DataFrame(meta_rows)

    return X, y, meta, feature_cols


class LagMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            # Funnel-style architecture:
            # input_dim -> 2 * hidden_dim -> hidden_dim -> output_dim
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_mean_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean R² across all output dimensions."""
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(r2_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(scores))


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    output_dim: int,
):
    """Train a small MLP on the lag features and return model and best val R²."""
    device = torch.device("cpu")

    model = LagMLP(input_dim=input_dim, hidden_dim=HIDDEN_SIZE, output_dim=output_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    loss_fn = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    best_val_r2 = -1e9
    best_state_dict = None

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb)
                val_preds.append(preds.cpu().numpy())
                val_targets.append(yb.numpy())

        val_preds_np = np.vstack(val_preds)
        val_targets_np = np.vstack(val_targets)
        val_r2 = compute_mean_r2(val_targets_np, val_preds_np)

        scheduler.step(val_r2)

        print(
            f"Epoch {epoch}/{N_EPOCHS} - "
            f"train_loss={train_loss:.6f}, val_mean_r2={val_r2:.6f}"
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state_dict = model.state_dict()

    # Load best-performing weights before returning
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, best_val_r2


def main() -> None:
    """
    Build supervised datasets, train a small MLP on lag features,
    report validation R², and save model + normalization parameters.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")

    print(f"Loading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)
    print(f"Full dataset shape: {df.shape}")

    df_train, df_val, train_ids, val_ids = split_by_seq(df)
    print(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")

    X_train, y_train, meta_train, feature_cols = build_supervised_dataset(
        df_train, n_lags=N_LAGS_DEFAULT
    )
    X_val, y_val, meta_val, _ = build_supervised_dataset(
        df_val, n_lags=N_LAGS_DEFAULT
    )

    print("\n=== Supervised dataset summary ===")
    print(f"State feature dim: {len(feature_cols)}")
    print(
        f"Train samples: {X_train.shape[0]}, X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
    )
    print(
        f"Val samples:   {X_val.shape[0]}, X_val shape: {X_val.shape}, y_val shape:   {y_val.shape}"
    )

    # Standardize X based on train statistics
    print("\nComputing normalization on train features...")
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8

    X_train_norm = (X_train - x_mean) / x_std
    X_val_norm = (X_val - x_mean) / x_std

    input_dim = X_train_norm.shape[1]
    output_dim = y_train.shape[1]

    print(
        f"Training MLP ensemble with input_dim={input_dim}, output_dim={output_dim}, "
        f"hidden_dim={HIDDEN_SIZE}, epochs={N_EPOCHS}, seeds={ENSEMBLE_SEEDS}"
    )

    # Directory for saving weights and normalization parameters
    weights_dir = os.path.join(base_dir, WEIGHTS_DIR)
    os.makedirs(weights_dir, exist_ok=True)

    best_overall_r2 = -1e9
    best_seed_idx: Optional[int] = None
    best_state_dict: Optional[dict] = None

    for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n=== Training ensemble member {seed_idx} with seed={seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model, best_val_r2 = train_mlp(
            X_train_norm, y_train, X_val_norm, y_val, input_dim, output_dim
        )

        print(
            f"Seed {seed} (index {seed_idx}) best validation mean R²: {best_val_r2:.6f}"
        )

        # Save this ensemble member
        seed_model_path = os.path.join(weights_dir, f"lag_mlp_seed{seed_idx}.pth")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": HIDDEN_SIZE,
                "n_lags": N_LAGS_DEFAULT,
            },
            seed_model_path,
        )
        print(f"Saved ensemble member {seed_idx} weights to: {seed_model_path}")

        if best_val_r2 > best_overall_r2:
            best_overall_r2 = best_val_r2
            best_seed_idx = seed_idx
            best_state_dict = model.state_dict()

    print(
        f"\nBest validation mean R² across ensemble members: {best_overall_r2:.6f} "
        f"(seed index {best_seed_idx})"
    )

    # Save a single best model checkpoint for backward compatibility
    model_path = os.path.join(weights_dir, "lag_mlp.pth")
    if best_state_dict is not None:
        torch.save(
            {
                "state_dict": best_state_dict,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": HIDDEN_SIZE,
                "n_lags": N_LAGS_DEFAULT,
            },
            model_path,
        )
        print(f"Saved best single-model weights to: {model_path}")

    # Save normalization parameters (shared by all ensemble members)
    norm_path = os.path.join(weights_dir, "lag_mlp_normalization.npz")
    model_path = os.path.join(weights_dir, "lag_mlp.pth")
    norm_path = os.path.join(weights_dir, "lag_mlp_normalization.npz")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": HIDDEN_SIZE,
            "n_lags": N_LAGS_DEFAULT,
        },
        model_path,
    )

    np.savez(
        norm_path,
        x_mean=x_mean,
        x_std=x_std,
        n_lags=N_LAGS_DEFAULT,
        feature_cols=np.array(feature_cols),
    )

    print(f"Saved normalization params to: {norm_path}")


if __name__ == "__main__":
    main()
