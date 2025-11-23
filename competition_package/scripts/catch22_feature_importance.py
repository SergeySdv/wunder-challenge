import os
from typing import Tuple

import numpy as np
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from train_model import (
    N_LAGS_DEFAULT,
    build_supervised_dataset,
    compute_mean_r2,
    load_dataset,
    split_by_seq,
)


def _train_catboost_with_catch22(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[CatBoostRegressor, float]:
    """
    Train a moderate CatBoost MultiRMSE model on lag+delta+rolling+step+catch22 features.

    We subsample the train set for speed; the goal is to get reasonable
    feature importances, not the best possible validation score.
    """
    rng = np.random.default_rng(42)
    # Subsample quite aggressively to keep runtime manageable; we only
    # need relative feature importances, not a state-of-the-art model.
    max_train = 30_000
    if X_train.shape[0] > max_train:
        indices = rng.choice(X_train.shape[0], size=max_train, replace=False)
        X_train_sub = X_train[indices]
        y_train_sub = y_train[indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    model = CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=20,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        thread_count=1,
        od_type="Iter",
        od_wait=10,
        verbose=20,
    )

    model.fit(
        X_train_sub,
        y_train_sub,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    val_predictions = model.predict(X_val)
    val_mean_r2 = compute_mean_r2(y_val, val_predictions)
    return model, val_mean_r2


def _analyze_catch22_importance(
    model: CatBoostRegressor,
    X_train: np.ndarray,
    feature_cols: list[str],
) -> None:
    """
    Decompose CatBoost feature importances into:
      - base (lag+delta+rolling+step) vs catch22,
      - per-catch22-stat and per-dimension contributions.
    """
    dim = len(feature_cols)
    n_lags = N_LAGS_DEFAULT

    # Base streaming feature block currently used in train_model.py:
    # - raw lags (n_lags * dim)
    # - LastKnown-delta (n_lags * dim)
    # - rolling mean/std (2 * dim)
    # - lag-1 autocorr (dim)
    # - fraction above mean (dim)
    # - step feature (1)
    base_dim = (2 * n_lags * dim) + (4 * dim) + 1
    total_dim = X_train.shape[1]
    catch22_dim = total_dim - base_dim

    if catch22_dim <= 0:
        raise ValueError(
            f"Expected positive catch22 feature block, got catch22_dim={catch22_dim} "
            f"(total_dim={total_dim}, base_dim={base_dim})"
        )

    if catch22_dim % dim != 0:
        raise ValueError(
            f"catch22_dim={catch22_dim} is not divisible by dim={dim}; "
            "unexpected catch22 layout."
        )

    n_stats = catch22_dim // dim

    importances = model.get_feature_importance()
    if importances.shape[0] != total_dim:
        raise ValueError(
            f"Feature importance length {importances.shape[0]} != total_dim {total_dim}"
        )

    base_importances = importances[:base_dim]
    catch22_importances = importances[base_dim:]

    total_importance = float(importances.sum())
    base_total = float(base_importances.sum())
    catch22_total = float(catch22_importances.sum())

    print("\n=== Global importance split ===")
    print(f"Total importance (all features): {total_importance:.4f}")
    print(
        f"Base v3 features (lags+delta+rolling+step): {base_total:.4f} "
        f"({100.0 * base_total / total_importance:.2f}%)"
    )
    print(
        f"Per-sequence catch22 block: {catch22_total:.4f} "
        f"({100.0 * catch22_total / total_importance:.2f}%)"
    )

    # Aggregate catch22 importance by original catch22 statistic and by dimension.
    seq_catch22 = np.load(
        os.path.join(os.path.dirname(__file__), "datasets", "catch22_per_seq.npz"),
        allow_pickle=True,
    )
    catch22_names = seq_catch22["catch22_names"].tolist()

    # Per-stat importance: aggregate over all 32 dimensions.
    stat_importance = np.zeros(n_stats, dtype=np.float64)
    for j in range(catch22_dim):
        stat_idx = j % n_stats
        stat_importance[stat_idx] += catch22_importances[j]

    # Normalize for readability
    stat_importance_pct = 100.0 * stat_importance / total_importance
    stat_order = np.argsort(stat_importance_pct)[::-1]

    print("\n=== Top catch22 statistics by importance (aggregated over dims) ===")
    for rank in range(min(10, n_stats)):
        s = stat_order[rank]
        name = catch22_names[s] if s < len(catch22_names) else f"idx_{s}"
        print(
            f"{rank+1:2d}. {name:30s}  "
            f"contribution ≈ {stat_importance_pct[s]:6.3f}% of total importance"
        )

    # Per-dimension importance: aggregate catch22 over the 22 stats for each of 32 dims.
    dim_importance = np.zeros(dim, dtype=np.float64)
    for d in range(dim):
        start = d * n_stats
        end = start + n_stats
        dim_importance[d] = catch22_importances[start:end].sum()

    dim_importance_pct = 100.0 * dim_importance / total_importance
    dim_order = np.argsort(dim_importance_pct)[::-1]

    print("\n=== Top state dimensions by catch22 importance ===")
    for rank in range(min(10, dim)):
        d = dim_order[rank]
        print(
            f"{rank+1:2d}. feature '{feature_cols[d]}'  "
            f"catch22 contribution ≈ {dim_importance_pct[d]:6.3f}% of total importance"
        )


def _cluster_sequences_in_catch22_space() -> None:
    """
    Cluster sequences in catch22 space to see high-level "types" of series.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    catch22_path = os.path.join(base_dir, "datasets", "catch22_per_seq.npz")

    if not os.path.exists(catch22_path):
        print(
            f"\nNo catch22_per_seq.npz found at {catch22_path}; "
            "skipping clustering step."
        )
        return

    data = np.load(catch22_path, allow_pickle=True)
    seq_ids = data["seq_ids"]
    catch22_values = data["catch22_values"]  # (n_seqs, n_dims, 22)

    n_seqs, n_dims, n_stats = catch22_values.shape
    flat = catch22_values.reshape(n_seqs, n_dims * n_stats).astype(np.float32)

    scaler = StandardScaler()
    flat_std = scaler.fit_transform(flat)

    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(flat_std)

    print("\n=== KMeans clustering in catch22 space ===")
    for cluster_idx in range(n_clusters):
        mask = labels == cluster_idx
        count = int(mask.sum())
        if count == 0:
            continue

        print(f"\nCluster {cluster_idx}: {count} sequences")

        # For a quick qualitative view, report a few simple aggregate stats:
        cluster_vals = flat[mask]
        # Approximate overall scale and variance in catch22 space
        mean_abs = float(np.mean(np.abs(cluster_vals)))
        std_vals = float(np.std(cluster_vals))
        print(f"  Mean |catch22|: {mean_abs:.4f}, Std of catch22: {std_vals:.4f}")

    print(
        "\n(For deeper analysis, you can export 'labels' per seq_ix and "
        "relate clusters to sequence-level errors or other metrics.)"
    )


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")

    print(f"Loading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)
    print(f"Full dataset shape: {df.shape}")

    df_train, df_val, train_ids, val_ids = split_by_seq(df)
    print(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")

    print("\nBuilding supervised datasets with catch22 features...")
    X_train, y_train, meta_train, feature_cols = build_supervised_dataset(
        df_train,
        n_lags=N_LAGS_DEFAULT,
        add_step_feature=True,
        use_catch22=True,
    )
    X_val, y_val, meta_val, _ = build_supervised_dataset(
        df_val,
        n_lags=N_LAGS_DEFAULT,
        add_step_feature=True,
        use_catch22=True,
    )

    print("\n=== Supervised dataset summary (with catch22) ===")
    print(f"State feature dim: {len(feature_cols)}")
    print(
        f"Train samples: {X_train.shape[0]}, X_train shape: {X_train.shape}, "
        f"y_train shape: {y_train.shape}"
    )
    print(
        f"Val samples:   {X_val.shape[0]}, X_val shape: {X_val.shape}, "
        f"y_val shape:   {y_val.shape}"
    )

    print("\nTraining CatBoost on lag+delta+rolling+step+catch22 features...")
    model, val_mean_r2 = _train_catboost_with_catch22(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    print(f"\nValidation mean R² (CatBoost with catch22): {val_mean_r2:.6f}")

    _analyze_catch22_importance(model, X_train, feature_cols)
    _cluster_sequences_in_catch22_space()


if __name__ == "__main__":
    main()
