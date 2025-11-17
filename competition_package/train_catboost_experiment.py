import os
import time

import numpy as np
from catboost import CatBoostRegressor

from train_model import (
    N_LAGS_DEFAULT,
    build_supervised_dataset,
    compute_mean_r2,
    load_dataset,
    split_by_seq,
)


def main() -> None:
    """
    Offline CatBoost experiment on the same lag+delta features
    as the LagMLP in train_model.py.

    This is intended as a Tsururu-like laboratory:
    - Reuses the supervised (X, y) built by build_supervised_dataset.
    - Trains a single multi-output CatBoostRegressor (MultiRMSE).
    - Reports validation mean R² and saves the model for further analysis.

    It is NOT used in the submission solution.py.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")

    print(f"Loading dataset from: {dataset_path}")
    dataset_df = load_dataset(dataset_path)
    print(f"Full dataset shape: {dataset_df.shape}")

    train_df, val_df, train_ids, val_ids = split_by_seq(dataset_df)
    print(f"Train sequences: {len(train_ids)}, Val sequences: {len(val_ids)}")

    X_train, y_train, meta_train, feature_cols = build_supervised_dataset(
        train_df, n_lags=N_LAGS_DEFAULT
    )
    X_val, y_val, meta_val, _ = build_supervised_dataset(
        val_df, n_lags=N_LAGS_DEFAULT
    )

    print("\n=== CatBoost supervised dataset summary ===")
    print(f"State feature dim: {len(feature_cols)}")
    print(
        f"Train samples: {X_train.shape[0]}, "
        f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
    )
    print(
        f"Val samples:   {X_val.shape[0]}, "
        f"X_val shape: {X_val.shape}, y_val shape:   {y_val.shape}"
    )

    # Full-data configuration ("let it cook"):
    # - use all supervised train samples
    # - more iterations, with early stopping to avoid wasting too much time
    model = CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        thread_count=1,
        od_type="Iter",
        od_wait=50,
        verbose=100,
    )

    print("\nTraining CatBoostRegressor (MultiRMSE) on lag+delta features...")
    start_time = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f} seconds.")

    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_mean_r2 = compute_mean_r2(y_val, val_predictions)
    print(f"Validation mean R² (CatBoost MultiRMSE): {val_mean_r2:.6f}")

    # Save model for potential reuse (offline analysis only)
    weights_dir = os.path.join(base_dir, "models")
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, "catboost_lag_delta_multiRMSE.cbm")
    model.save_model(model_path)
    print(f"Saved CatBoost model to: {model_path}")


if __name__ == "__main__":
    main()
