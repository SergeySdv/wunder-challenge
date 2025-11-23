"""
Quick leak probe for train.parquet.

This script looks for trivial patterns of the form:
    state[t+1, j] == state[t, j]
for each feature j, across all sequences.

If any feature has an equality ratio very close to 1.0, it may indicate
a trivial "next value equals current value" relationship that could be
exploited or might signal a deeper data issue.

It does NOT prove the absence of more subtle leaks; it is just a sanity check.
"""

import os

import numpy as np

from train_model import load_dataset


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")

    print(f"Loading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)
    print(f"Full dataset shape: {df.shape}")

    feature_cols = [
        c
        for c in df.columns
        if c not in ("seq_ix", "step_in_seq", "need_prediction")
    ]
    dim = len(feature_cols)
    print(f"Checking {dim} feature columns: {feature_cols}")

    same_match_counts = np.zeros(dim, dtype=np.int64)
    same_total_counts = np.zeros(dim, dtype=np.int64)

    for seq_ix, df_seq in df.groupby("seq_ix"):
        df_seq = df_seq.sort_values("step_in_seq")
        values = df_seq[feature_cols].values.astype(np.float64)  # (T, dim)
        if values.shape[0] < 2:
            continue

        x = values[:-1, :]  # at time t
        y = values[1:, :]  # at time t+1

        # Equalities for same feature j: y[:, j] == x[:, j]
        same_eq = (y == x)  # shape (T-1, dim)
        same_match_counts += same_eq.sum(axis=0)
        same_total_counts += same_eq.shape[0]

    print("\n=== Same-feature next-step equality ratios (y[t+1, j] == y[t, j]) ===")
    ratios = same_match_counts / same_total_counts
    for j, col in enumerate(feature_cols):
        print(
            f"Feature {col:>2}: matches={same_match_counts[j]}, "
            f"total={same_total_counts[j]}, ratio={ratios[j]:.6f}"
        )

    max_idx = int(np.argmax(ratios))
    print(
        f"\nMax same-feature equality ratio: feature {feature_cols[max_idx]} "
        f"with ratio={ratios[max_idx]:.6f}"
    )
    print(
        "\nIf any ratio is extremely close to 1.0, that could indicate a trivial "
        "leak pattern. Here we only checked the simplest case (same feature, "
        "next step vs current step); more complex leaks would require deeper analysis."
    )


if __name__ == "__main__":
    main()

