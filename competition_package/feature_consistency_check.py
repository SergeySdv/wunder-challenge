import os
from typing import List

import numpy as np

from train_model import (
    N_LAGS_DEFAULT,
    build_supervised_dataset,
    load_dataset,
)
from utils import DataPoint
from solution import PredictionModel


def main() -> None:
    """
    Sanity check: verify that the normalized feature vectors built offline
    by build_supervised_dataset + train_model normalization match the
    normalized vectors built online inside PredictionModel._build_features.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")
    norm_path = os.path.join(base_dir, "models", "lag_mlp_normalization.npz")

    print(f"Loading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)
    print(f"Full dataset shape: {df.shape}")

    print("Building supervised dataset (offline features)...")
    X, y, meta, feature_cols = build_supervised_dataset(
        df, n_lags=N_LAGS_DEFAULT, add_step_feature=True, use_catch22=False
    )
    print(f"Supervised X shape: {X.shape}, y shape: {y.shape}")

    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Normalization file not found at {norm_path}. "
            "Run train_model.py first."
        )

    norm = np.load(norm_path, allow_pickle=True)
    x_mean = norm["x_mean"].astype(np.float32)
    x_std = norm["x_std"].astype(np.float32)

    # Normalized offline features
    X_norm_offline = (X - x_mean) / x_std

    # Prepare mapping (seq_ix -> rows) for streaming replay
    grouped = {
        seq_ix: g.sort_values("step_in_seq")
        for seq_ix, g in df.groupby("seq_ix")
    }

    # Choose a small random subset of samples to compare
    rng = np.random.default_rng(0)
    n_checks = 50
    indices: List[int] = rng.choice(len(meta), size=n_checks, replace=False).tolist()

    print(f"Checking {n_checks} random samples for feature consistency...")
    model = PredictionModel()

    n_mismatches = 0
    for idx in indices:
        row_meta = meta.iloc[idx]
        seq_ix = int(row_meta["seq_ix"])
        current_step = int(row_meta["current_step"])

        df_seq = grouped[seq_ix]
        states = df_seq[feature_cols].values
        steps = df_seq["step_in_seq"].values
        needs = df_seq["need_prediction"].values

        # Replay sequence up to current_step through PredictionModel
        model._reset_sequence(seq_ix)
        last_dp = None
        for s, step_in_seq, need_pred in zip(states, steps, needs):
            step_int = int(step_in_seq)
            dp = DataPoint(
                seq_ix=seq_ix,
                step_in_seq=step_int,
                need_prediction=bool(need_pred),
                state=s.astype(np.float32),
            )
            # PredictionModel manages state_history inside predict()
            _ = model.predict(dp)
            if step_int == current_step:
                last_dp = dp
                break

        if last_dp is None:
            raise RuntimeError(
                f"Could not find step_in_seq={current_step} for seq_ix={seq_ix}"
            )

        # Build online normalized features for this point
        x_norm_online = model._build_features(last_dp)
        if x_norm_online is None:
            raise RuntimeError(
                f"Not enough history for seq_ix={seq_ix}, step={current_step} "
                "in consistency check (this should not happen)."
            )

        x_norm_offline = X_norm_offline[idx]

        if not np.allclose(x_norm_online, x_norm_offline, atol=1e-5):
            n_mismatches += 1
            max_diff = float(np.max(np.abs(x_norm_online - x_norm_offline)))
            print(
                f"Mismatch for sample {idx} (seq_ix={seq_ix}, step={current_step}), "
                f"max |Δ|={max_diff:.3e}"
            )

    if n_mismatches == 0:
        print("✓ All checked samples matched between offline and online features.")
    else:
        print(
            f"⚠ {n_mismatches} / {n_checks} samples had mismatched features. "
            "Inspect logs above for details."
        )


if __name__ == "__main__":
    main()

