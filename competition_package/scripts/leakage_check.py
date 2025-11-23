"""
Leakage checking script.

This script verifies that our supervised feature builder:
  - Uses only past information (lagged states) to predict the next state.
  - Does not mix sequences between train and validation splits.

It does NOT train a model; it only reconstructs the raw data around a
subset of samples and checks that:
  - X lag windows match the last N_LAGS_DEFAULT states up to current_step.
  - y matches the state at target_step = current_step + 1.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

from train_model import (
    N_LAGS_DEFAULT,
    build_supervised_dataset,
    load_dataset,
    split_by_seq,
)


def _check_no_seq_overlap(train_ids: set, val_ids: set) -> None:
    overlap = train_ids & val_ids
    if overlap:
        raise AssertionError(f"Train/val seq_ix overlap detected: {sorted(overlap)[:10]} ...")


def _check_temporal_consistency(
    df_split: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    feature_cols: list,
    n_lags: int,
    label: str,
    max_checks: int = 200,
) -> None:
    """
    For a random subset of samples, verify that:
      - target_step == current_step + 1
      - X lag window equals the last n_lags states up to current_step
      - y equals the state at target_step
    """
    if len(meta) == 0:
        print(f"[{label}] No samples to check (meta empty).")
        return

    dim = len(feature_cols)
    lag_dim = dim * n_lags

    rng = np.random.default_rng(0)
    n_checks = min(max_checks, len(meta))
    indices = rng.choice(len(meta), size=n_checks, replace=False)

    # Pre-split by sequence for faster access
    grouped = {
        seq_ix: g.sort_values("step_in_seq")
        for seq_ix, g in df_split.groupby("seq_ix")
    }

    for i in indices:
        row_meta = meta.iloc[i]
        seq_ix = row_meta["seq_ix"]
        current_step = row_meta["current_step"]
        target_step = row_meta["target_step"]

        if target_step != current_step + 1:
            raise AssertionError(
                f"[{label}] target_step != current_step + 1 for sample {i}: "
                f"{target_step} != {current_step} + 1"
            )

        df_seq = grouped[int(seq_ix)]
        states = df_seq[feature_cols].values
        steps = df_seq["step_in_seq"].values

        # find index of current_step within this sequence
        idx_seq_array = np.nonzero(steps == current_step)[0]
        if len(idx_seq_array) != 1:
            raise AssertionError(
                f"[{label}] Expected exactly one index for (seq_ix={seq_ix}, step={current_step}), "
                f"found {len(idx_seq_array)}"
            )
        idx_seq = int(idx_seq_array[0])

        # reconstruct expected lag window and target
        if idx_seq < n_lags - 1:
            raise AssertionError(
                f"[{label}] Sample {i} uses idx_seq={idx_seq} < n_lags-1={n_lags-1}, "
                "but supervised builder should have skipped it."
            )

        lag_slice = states[idx_seq - n_lags + 1 : idx_seq + 1]  # (n_lags, dim)
        lag_flat_expected = lag_slice.reshape(-1)
        target_expected = states[idx_seq + 1]

        lag_flat_actual = X[i, :lag_dim]
        target_actual = y[i]

        if not np.allclose(lag_flat_actual, lag_flat_expected):
            raise AssertionError(
                f"[{label}] Lag window mismatch for sample {i} (seq_ix={seq_ix}, step={current_step})"
            )
        if not np.allclose(target_actual, target_expected):
            raise AssertionError(
                f"[{label}] Target mismatch for sample {i} (seq_ix={seq_ix}, step={current_step})"
            )


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")

    print(f"Loading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)
    print(f"Full dataset shape: {df.shape}")

    df_train, df_val, train_ids, val_ids = split_by_seq(df)
    print(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")

    _check_no_seq_overlap(train_ids, val_ids)
    print("✓ No overlap between train and validation seq_ix sets.")

    print("\nBuilding supervised datasets (this may take a bit)...")
    X_train, y_train, meta_train, feature_cols = build_supervised_dataset(
        df_train, n_lags=N_LAGS_DEFAULT
    )
    X_val, y_val, meta_val, _ = build_supervised_dataset(
        df_val, n_lags=N_LAGS_DEFAULT
    )

    print(f"Train supervised shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val supervised shape:   X={X_val.shape}, y={y_val.shape}")

    print("\nChecking temporal consistency for a subset of samples...")
    _check_temporal_consistency(
        df_split=df_train,
        X=X_train,
        y=y_train,
        meta=meta_train,
        feature_cols=feature_cols,
        n_lags=N_LAGS_DEFAULT,
        label="train",
    )
    _check_temporal_consistency(
        df_split=df_val,
        X=X_val,
        y=y_val,
        meta=meta_val,
        feature_cols=feature_cols,
        n_lags=N_LAGS_DEFAULT,
        label="val",
    )

    print("\nAll leakage checks passed: no future information is used in features, and train/val are sequence-disjoint.")


if __name__ == "__main__":
    main()

