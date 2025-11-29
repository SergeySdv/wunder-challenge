"""
Micro-Mamba (SSD-style) experiment on raw 32-dim windows (sequence-to-one).
CPU-friendly, pure PyTorch, no external kernels.
"""

import argparse
import os
import random
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from src.features.extractor import FeatureExtractor, feature_dim


# -----------------------------
# Config / Defaults
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
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


def build_supervised_dataset(
    df: pd.DataFrame,
    clip_min: np.ndarray,
    clip_max: np.ndarray,
    window: int,
    extractor: FeatureExtractor,
    feature_mode: str = "v19",
    residual_target: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return X of shape (N, feature_dim) and y of shape (N, 32).
    If residual_target=True, targets are y_{t+1} - x_t (delta); else absolute next value.
    """
    feature_cols = [str(i) for i in range(32)]
    X_list, y_list = [], []

    for _, df_seq in df.groupby("seq_ix"):
        states = df_seq[feature_cols].values
        need_pred = df_seq["need_prediction"].values
        T = len(df_seq)
        if T < window + 1:
            continue
        for idx in range(window - 1, T - 1):
            if not need_pred[idx]:
                continue
            window_slice = states[idx - window + 1 : idx + 1]
            window_slice = np.clip(window_slice, clip_min, clip_max).astype(np.float32)
            step_in_seq = int(df_seq["step_in_seq"].iloc[idx])
            if feature_mode == "minimal":
                last = window_slice[-1]
                lag_flat = window_slice.reshape(-1)
                delta_flat = (window_slice - last).reshape(-1)
                step_val = np.array([step_in_seq / 1000.0], dtype=np.float32)
                features = np.concatenate([lag_flat, delta_flat, step_val]).astype(np.float32)
            else:
                features = extractor.build_window_features(window_slice, step_in_seq)
            target_raw = states[idx + 1].astype(np.float32)
            if residual_target:
                target = target_raw - window_slice[-1]
            else:
                target = target_raw
            X_list.append(features)
            y_list.append(target)

    if not X_list:
        if feature_mode == "minimal":
            dim = window * 32 * 2 + 1
        else:
            dim = feature_dim(window)
        return np.empty((0, dim), dtype=np.float32), np.empty((0, 32), dtype=np.float32)

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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class MambaBlock(nn.Module):
    """
    Simplified SSD-style block:
    - Depthwise conv for local context
    - Diagonal/elementwise decay per head
    - Gated mix of SSM output and conv output
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 32,
        nheads: int = 4,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nheads == 0, "d_model must be divisible by nheads"
        self.d_model = d_model
        self.nheads = nheads
        self.headdim = d_model // nheads
        self.d_state = d_state

        padding = d_conv - 1
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=padding,
            groups=d_model,
        )
        self.conv_activation = nn.SiLU()

        # Projections
        self.in_proj = nn.Linear(d_model, d_model * 3)  # for value, gate, skip
        self.u_proj = nn.Linear(d_model, nheads * d_state)  # state update input
        self.c_proj = nn.Linear(d_model, nheads * d_state)  # state-to-output filter
        self.dt_proj = nn.Linear(d_model, nheads)  # timescale/decay modulator

        # State decay parameters (per head)
        self.A_log = nn.Parameter(torch.randn(nheads, d_state) * -0.3)  # negative for stability
        self.dt_bias = nn.Parameter(torch.zeros(nheads))

        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # FFN
        ffn_hidden = int(d_model * 2)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, x):
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)
        """
        B, T, D = x.shape
        # Depthwise conv over time
        x_conv = self.depthwise_conv(x.transpose(1, 2))  # (B, D, T)
        x_conv = x_conv[:, :, :T]  # causal trim
        x_conv = self.conv_activation(x_conv).transpose(1, 2)  # (B, T, D)

        # Projections
        proj = self.in_proj(x)  # (B, T, 3D)
        v, gate_raw, skip = torch.split(proj, D, dim=-1)
        gate = torch.sigmoid(gate_raw)

        u = self.u_proj(x).view(B, T, self.nheads, self.d_state)  # (B, T, H, S)
        c = self.c_proj(x).view(B, T, self.nheads, self.d_state)  # (B, T, H, S)
        dt = torch.nn.functional.softplus(self.dt_proj(x) + self.dt_bias)  # (B, T, H)

        # Init state
        state = x.new_zeros(B, self.nheads, self.d_state)
        A = -torch.exp(self.A_log)  # (H, S), negative

        outputs = []
        for t in range(T):
            decay = torch.exp(A * dt[:, t].unsqueeze(-1))  # (B, H, S)
            state = state * decay + u[:, t] * dt[:, t].unsqueeze(-1)  # accumulate
            # state to head outputs
            h_t = (state * c[:, t]).sum(dim=-1)  # (B, H)
            h_t = h_t.unsqueeze(-1).repeat(1, 1, self.headdim)  # (B, H, P)
            h_t = h_t.reshape(B, self.d_model)  # (B, D)

            # gated mix with conv features
            out_t = gate[:, t] * h_t + (1 - gate[:, t]) * x_conv[:, t] + skip[:, t]
            outputs.append(out_t.unsqueeze(1))

        y = torch.cat(outputs, dim=1)  # (B, T, D)
        y = self.norm(y)
        y = y + self.dropout(self.ffn(y))
        return y


class MicroMambaModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 32,
        d_model: int = 64,
        d_state: int = 32,
        nheads: int = 4,
        n_layers: int = 2,
        d_conv: int = 4,
        dropout: float = 0.1,
        residual_output: bool = True,
    ):
        super().__init__()
        self.residual_output = residual_output
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(d_model=d_model, d_state=d_state, nheads=nheads, d_conv=d_conv, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 32)  # predict 32-dim

    def forward(self, x):
        """
        x: (B, T, 32)
        returns: (B, 32) prediction for next step
        """
        h = self.embed(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h)
        h_last = h[:, -1]  # use last timestep representation
        out = self.head(h_last)
        return out


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

    if not all_preds:
        return float("nan"), float("nan")

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    r2s = [r2_score(all_targets[:, i], all_preds[:, i]) for i in range(32)]
    return total_loss / len(dataloader.dataset), float(np.mean(r2s))


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Micro-Mamba (SSD-style) on raw windows.")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Sequence length for input window.")
    parser.add_argument("--d_model", type=int, default=64, help="Model width.")
    parser.add_argument("--d_state", type=int, default=32, help="State dim per head.")
    parser.add_argument("--nheads", type=int, default=4, help="Number of heads.")
    parser.add_argument("--layers", type=int, default=2, help="Number of SSM blocks.")
    parser.add_argument("--d_conv", type=int, default=4, help="Conv kernel size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay.")
    parser.add_argument("--subset", type=int, default=None, help="Optional cap on training samples for quick tests.")
    parser.add_argument("--residual_target", action="store_true", help="Use residual target (y - last).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device.")
    parser.add_argument("--save_model_path", type=str, default=None, help="Path to save model state_dict.")
    parser.add_argument("--save_norm_path", type=str, default=None, help="Path to save normalization (npz).")
    parser.add_argument("--save_meta_path", type=str, default=None, help="Path to save meta (json).")
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="v19",
        choices=["v19", "minimal"],
        help="Feature set: v19 (full engineered) or minimal (lags+deltas+step).",
    )
    args = parser.parse_args()

    set_seed(2024)

    print(
        f"--- Micro-Mamba (window={args.window}, d_model={args.d_model}, d_state={args.d_state}, "
        f"layers={args.layers}, residual_target={args.residual_target}) ---"
    )
    root_dir = os.path.abspath(os.path.join(BASE_DIR, ".."))
    dataset_path = os.path.join(root_dir, "datasets", "train.parquet")
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
    extractor = FeatureExtractor(n_lags=args.window, clip_min=clip_min, clip_max=clip_max)

    print("Building train dataset...")
    X_train, y_train = build_supervised_dataset(
        df_train,
        clip_min,
        clip_max,
        args.window,
        extractor=extractor,
        feature_mode=args.feature_mode,
        residual_target=args.residual_target,
    )
    if args.subset is not None and args.subset < len(X_train):
        idx = np.random.choice(len(X_train), args.subset, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    print(f"  Train samples: {len(X_train)}")

    print("Building val dataset...")
    X_val, y_val = build_supervised_dataset(
        df_val,
        clip_min,
        clip_max,
        args.window,
        extractor=extractor,
        feature_mode=args.feature_mode,
        residual_target=args.residual_target,
    )
    print(f"  Val samples: {len(X_val)}")

    print("Building pseudo-LB dataset...")
    X_pseudo, y_pseudo = build_supervised_dataset(
        df_pseudo,
        clip_min,
        clip_max,
        args.window,
        extractor=extractor,
        feature_mode=args.feature_mode,
        residual_target=args.residual_target,
    )
    print(f"  Pseudo-LB samples: {len(X_pseudo)}")

    if len(X_train) == 0:
        raise RuntimeError("No training samples found. Check need_prediction filtering or window size.")

    print("Computing normalization stats on train...")
    mean, std = compute_norm_stats(X_train)
    X_train = normalize_windows(X_train, mean, std)
    X_val = normalize_windows(X_val, mean, std) if len(X_val) else X_val
    X_pseudo = normalize_windows(X_pseudo, mean, std) if len(X_pseudo) else X_pseudo

    # Add a time dimension of 1 for the Mamba blocks
    X_train = X_train[:, None, :]
    X_val = X_val[:, None, :] if len(X_val) else X_val
    X_pseudo = X_pseudo[:, None, :] if len(X_pseudo) else X_pseudo

    train_ds = WindowDataset(X_train, y_train)
    val_ds = WindowDataset(X_val, y_val) if len(X_val) else None
    pseudo_ds = WindowDataset(X_pseudo, y_pseudo) if len(X_pseudo) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True) if val_ds else None
    pseudo_loader = DataLoader(pseudo_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True) if pseudo_ds else None

    device = torch.device(args.device)
    model = MicroMambaModel(
        input_dim=X_train.shape[-1],
        d_model=args.d_model,
        d_state=args.d_state,
        nheads=args.nheads,
        n_layers=args.layers,
        d_conv=args.d_conv,
        dropout=args.dropout,
        residual_output=args.residual_target,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_r2 = -np.inf
    best_state = None
    patience = 5
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        if val_loader:
            val_loss, val_r2 = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_r2 = float("nan"), float("nan")

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | val_r2={val_r2:.5f}"
        )

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

    # Save artifacts
    if args.save_model_path:
        torch.save(model.state_dict(), args.save_model_path)
        print(f"Saved model to {args.save_model_path}")
    if args.save_norm_path:
        np.savez(
            args.save_norm_path,
            mean=mean,
            std=std,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        print(f"Saved normalization to {args.save_norm_path}")
    if args.save_meta_path:
        import json

        meta = dict(
            window=args.window,
            input_dim=X_train.shape[-1],
            d_model=args.d_model,
            d_state=args.d_state,
            nheads=args.nheads,
            layers=args.layers,
            d_conv=args.d_conv,
            dropout=args.dropout,
            residual_target=args.residual_target,
            weight_decay=args.weight_decay,
            lr=args.lr,
            feature_mode=args.feature_mode,
        )
        with open(args.save_meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta to {args.save_meta_path}")
    else:
        print("Pseudo-LB set empty; skipping.")


if __name__ == "__main__":
    main()
