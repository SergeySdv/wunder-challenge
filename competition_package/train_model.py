import os
from typing import Tuple, List, Set

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import nn


N_LAGS_DEFAULT = 10
HIDDEN_SIZE = 64
N_EPOCHS = 10
BATCH_SIZE = 1024
LR = 1e-3
WEIGHTS_DIR = "models"


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load and sort the competition dataset."""
    df = pd.read_parquet(dataset_path)
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    return df


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
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Build a supervised learning dataset from the raw table.

    For each sequence and each step t where:
      - need_prediction == 1
      - there are at least `n_lags` past observations
      - there is a next step within the same sequence

    we create:
      - X_t: concatenation of the last `n_lags` state vectors up to step t,
             optionally plus a simple normalized step feature.
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

            features = [lag_flat, delta_flat, mean_last, std_last]
            if add_step_feature:
                # step position as a simple normalized scalar
                features.append(np.array([current_step / 1000.0], dtype=np.float32))

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
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
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

        print(
            f"Epoch {epoch}/{N_EPOCHS} - "
            f"train_loss={train_loss:.6f}, val_mean_r2={val_r2:.6f}"
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2

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
        f"Training MLP with input_dim={input_dim}, output_dim={output_dim}, "
        f"hidden_dim={HIDDEN_SIZE}, epochs={N_EPOCHS}"
    )
    model, best_val_r2 = train_mlp(
        X_train_norm, y_train, X_val_norm, y_val, input_dim, output_dim
    )

    print(f"\nBest validation mean R²: {best_val_r2:.6f}")

    # Save model and normalization parameters for later use in solution.py
    weights_dir = os.path.join(base_dir, WEIGHTS_DIR)
    os.makedirs(weights_dir, exist_ok=True)

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

    print(f"Saved model weights to: {model_path}")
    print(f"Saved normalization params to: {norm_path}")


if __name__ == "__main__":
    main()
