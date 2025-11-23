import argparse
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Config / Defaults
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WINDOW = 30
PSEUDO_LB_SEED = 999
VAL_SEED = 123
VAL_FRACTION = 0.1  # fraction of dev sequences held out for val


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Data utilities
# -----------------------------
def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    return df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)


def compute_winsorization_bounds(df: pd.DataFrame, lower_q=0.001, upper_q=0.999) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = [str(i) for i in range(32)]
    data = df[feature_cols].values
    lower = np.quantile(data, lower_q, axis=0).astype(np.float32)
    upper = np.quantile(data, upper_q, axis=0).astype(np.float32)
    return lower, upper


def build_supervised_dataset(df: pd.DataFrame, clip_min: np.ndarray, clip_max: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return X of shape (N, window, 32) and y of shape (N, 32)."""
    feature_cols = [str(i) for i in range(32)]
    X_list, y_list = [], []

    for _, df_seq in df.groupby("seq_ix"):
        states = df_seq[feature_cols].values
        need_pred = df_seq["need_prediction"].values
        T = len(df_seq)
        if T < window + 1:
            continue
        # slide over sequence; window ends at idx, target is idx+1
        for idx in range(window - 1, T - 1):
            if not need_pred[idx]:
                continue
            window_slice = states[idx - window + 1 : idx + 1]
            window_slice = np.clip(window_slice, clip_min, clip_max).astype(np.float32)
            target = states[idx + 1].astype(np.float32)
            X_list.append(window_slice)
            y_list.append(target)

    if not X_list:
        return np.empty((0, window, 32), dtype=np.float32), np.empty((0, 32), dtype=np.float32)

    return np.stack(X_list), np.stack(y_list)


def compute_norm_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std across all time positions."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32) + 1e-8
    return mean, std


def normalize_windows(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


# -----------------------------
# Dataset / Model
# -----------------------------
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # (batch, hidden)
        return self.head(h_last)


# -----------------------------
# Training / Eval
# -----------------------------
def run_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        total_loss += loss.item() * len(xb)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0, 32))
    all_targets = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0, 32))

    if len(all_preds) == 0:
        return float("nan"), float("nan")

    r2s = [r2_score(all_targets[:, i], all_preds[:, i]) for i in range(32)]
    return total_loss / len(dataloader.dataset), float(np.mean(r2s))


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="LSTM baseline on raw windows (sequence-to-one).")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Sequence length for input window.")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size of LSTM.")
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for LSTM/head.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--subset", type=int, default=None, help="Optional cap on training samples for quick tests.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device.")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save trained model state_dict (.pth).")
    parser.add_argument("--save_norm", type=str, default=None, help="Optional path to save norm/clip stats (.npz).")
    parser.add_argument("--save_meta", type=str, default=None, help="Optional path to save meta json (window, hidden, layers).")
    args = parser.parse_args()

    set_seed(2024)

    print(f"--- LSTM (window={args.window}, hidden={args.hidden}, layers={args.layers}) ---")
    dataset_path = os.path.join(BASE_DIR, "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    all_seqs = df["seq_ix"].unique()

    # Pseudo-LB split
    rng = np.random.default_rng(PSEUDO_LB_SEED)
    rng.shuffle(all_seqs)
    n_pseudo = int(len(all_seqs) * 0.10)
    pseudo_ids = all_seqs[:n_pseudo]
    dev_ids = all_seqs[n_pseudo:]

    # Train/Val split inside dev
    rng_val = np.random.default_rng(VAL_SEED)
    rng_val.shuffle(dev_ids)
    n_val = max(1, int(len(dev_ids) * VAL_FRACTION))
    val_ids = dev_ids[:n_val]
    train_ids = dev_ids[n_val:]

    df_train = df[df["seq_ix"].isin(train_ids)].copy()
    df_val = df[df["seq_ix"].isin(val_ids)].copy()
    df_pseudo = df[df["seq_ix"].isin(pseudo_ids)].copy()

    print("Computing winsorization bounds on train split...")
    clip_min, clip_max = compute_winsorization_bounds(df_train)

    print("Building train dataset...")
    X_train, y_train = build_supervised_dataset(df_train, clip_min, clip_max, args.window)
    if args.subset is not None and args.subset < len(X_train):
        idx = np.random.choice(len(X_train), args.subset, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    print(f"  Train samples: {len(X_train)}")

    print("Building val dataset...")
    X_val, y_val = build_supervised_dataset(df_val, clip_min, clip_max, args.window)
    print(f"  Val samples: {len(X_val)}")

    print("Building pseudo-LB dataset...")
    X_pseudo, y_pseudo = build_supervised_dataset(df_pseudo, clip_min, clip_max, args.window)
    print(f"  Pseudo-LB samples: {len(X_pseudo)}")

    if len(X_train) == 0:
        raise RuntimeError("No training samples found. Check need_prediction filtering or window size.")

    print("Computing normalization stats on train...")
    mean, std = compute_norm_stats(X_train)
    X_train = normalize_windows(X_train, mean, std)
    X_val = normalize_windows(X_val, mean, std) if len(X_val) else X_val
    X_pseudo = normalize_windows(X_pseudo, mean, std) if len(X_pseudo) else X_pseudo

    train_ds = WindowDataset(X_train, y_train)
    val_ds = WindowDataset(X_val, y_val) if len(X_val) else None
    pseudo_ds = WindowDataset(X_pseudo, y_pseudo) if len(X_pseudo) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True) if val_ds else None
    pseudo_loader = DataLoader(pseudo_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True) if pseudo_ds else None

    device = torch.device(args.device)
    model = LSTMRegressor(input_dim=32, hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_r2 = -np.inf
    best_state = None
    patience = 3
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        if val_loader:
            val_loss, val_r2 = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_r2 = float("nan"), float("nan")

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | val_r2={val_r2:.5f}")

        if val_loader and val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = model.state_dict()
            patience_ctr = 0
        elif val_loader:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval
    if val_loader:
        val_loss, val_r2 = evaluate(model, val_loader, criterion, device)
        print(f"Final Val R2: {val_r2:.5f}")

    if pseudo_loader:
        pseudo_loss, pseudo_r2 = evaluate(model, pseudo_loader, criterion, device)
        print(f"Pseudo-LB R2: {pseudo_r2:.5f}")
    else:
        print("Pseudo-LB set empty; skipping.")

    # Save artifacts if requested
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved model to {args.save_path}")
    if args.save_norm:
        np.savez(
            args.save_norm,
            mean=mean,
            std=std,
            clip_min=clip_min,
            clip_max=clip_max,
            window=args.window,
            hidden=args.hidden,
            layers=args.layers,
        )
        print(f"Saved normalization stats to {args.save_norm}")
    if args.save_meta:
        import json

        meta = {
            "window": args.window,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
        }
        with open(args.save_meta, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta to {args.save_meta}")


if __name__ == "__main__":
    main()
