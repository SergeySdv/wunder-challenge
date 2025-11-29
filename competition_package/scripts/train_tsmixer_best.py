import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Add project root folder to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from src.models.torchtsmixer import TSMixer

# --- Best Params from Optuna Trial 1 ---
# Value: 0.3803
# Params: use_deltas=True, num_blocks=2, ff_dim=256, dropout=0.17, lr=2.7e-4
USE_DELTAS = True
NUM_BLOCKS = 2
FF_DIM = 256
DROPOUT = 0.17
LR = 2.7e-4

N_LAGS = 10
BATCH_SIZE = 512
N_EPOCHS = 20
WEIGHTS_DIR = "models"
SAVE_PREFIX = "tsmixer_v2_delta"
CV_FOLDS = 5
CV_SEED = 42
PSEUDO_LB_SEED = 999

def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    return df

def compute_winsorization_bounds(df: pd.DataFrame, lower_q=0.001, upper_q=0.999):
    feature_cols = [str(i) for i in range(32)]
    data = df[feature_cols].values
    lower = np.quantile(data, lower_q, axis=0).astype(np.float32)
    upper = np.quantile(data, upper_q, axis=0).astype(np.float32)
    return lower, upper

def build_dataset_with_deltas(df: pd.DataFrame, clip_min: np.ndarray, clip_max: np.ndarray, use_deltas: bool = True):
    feature_cols = [str(i) for i in range(32)]
    X_list = []
    y_list = []
    
    for seq_ix, df_seq in df.groupby("seq_ix"):
        df_seq = df_seq.sort_values("step_in_seq")
        states = df_seq[feature_cols].values
        need_pred = df_seq["need_prediction"].values
        
        T = len(df_seq)
        for idx in range(T - 1):
            if not need_pred[idx]:
                continue
            if idx < N_LAGS - 1:
                continue
            
            # Raw lag window: (10, 32)
            raw_slice = states[idx - N_LAGS + 1 : idx + 1]
            lag_slice = np.clip(raw_slice, clip_min, clip_max).astype(np.float32)
            
            if use_deltas:
                # Compute diffs along time axis
                # For t=0, we duplicate the first value or use 0?
                # Let's use standard diff: out[i] = a[i+1] - a[i]
                # Here we want diff[t] = lag[t] - lag[t-1]
                diffs = np.zeros_like(lag_slice)
                diffs[1:] = lag_slice[1:] - lag_slice[:-1]
                # For index 0, let's just keep 0 (no info)
                
                combined = np.concatenate([lag_slice, diffs], axis=1)
                X_list.append(combined)
            else:
                X_list.append(lag_slice)
                
            y_list.append(states[idx+1].astype(np.float32))
            
    return np.array(X_list), np.array(y_list)

def train_one_fold(X_train, y_train, X_val, y_val, fold_idx):
    device = torch.device("cpu")
    input_channels = 64 if USE_DELTAS else 32
    
    model = TSMixer(
        sequence_length=N_LAGS,
        prediction_length=1,
        input_channels=input_channels,
        output_channels=32,
        num_blocks=NUM_BLOCKS,
        ff_dim=FF_DIM,
        dropout_rate=DROPOUT,
        activation_fn="relu",
        normalize_before=True,
        norm_type="batch"
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    loss_fn = nn.MSELoss()
    
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    best_val_r2 = -1e9
    best_state = None
    
    print(f"  [Fold {fold_idx}] Training...")
    
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(1)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).squeeze(1)
                preds.append(out.cpu().numpy())
                targets.append(yb.numpy())
        
        preds = np.vstack(preds)
        targets = np.vstack(targets)
        
        r2s = [r2_score(targets[:, i], preds[:, i]) for i in range(targets.shape[1])]
        mean_r2 = np.mean(r2s)
        scheduler.step(mean_r2)
        
        if mean_r2 > best_val_r2:
            best_val_r2 = mean_r2
            best_state = model.state_dict()
            
    print(f"  [Fold {fold_idx}] Best Val R2: {best_val_r2:.5f}")
    return best_state, best_val_r2

def main():
    print(f"--- TSMixer v2 (Best Optuna Params) ---")
    print(f"USE_DELTAS={USE_DELTAS}, BLOCKS={NUM_BLOCKS}, FF_DIM={FF_DIM}, DO={DROPOUT}, LR={LR}")
    
    dataset_path = os.path.join(CURRENT_DIR, "..", "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    
    all_seqs = df["seq_ix"].unique()
    rng = np.random.default_rng(PSEUDO_LB_SEED)
    rng.shuffle(all_seqs)
    
    n_pseudo = int(len(all_seqs) * 0.10)
    pseudo_ids = all_seqs[:n_pseudo]
    dev_ids = all_seqs[n_pseudo:]
    
    df_dev = df[df["seq_ix"].isin(dev_ids)].copy()
    df_pseudo = df[df["seq_ix"].isin(pseudo_ids)].copy()
    
    print("Computing Winsorization bounds...")
    clip_min, clip_max = compute_winsorization_bounds(df_dev)
    
    print("Building datasets...")
    X_dev, y_dev = build_dataset_with_deltas(df_dev, clip_min, clip_max, use_deltas=USE_DELTAS)
    X_pseudo, y_pseudo = build_dataset_with_deltas(df_pseudo, clip_min, clip_max, use_deltas=USE_DELTAS)
    
    print(f"Input shape: {X_dev.shape}")
    input_channels = X_dev.shape[2]
    
    # Normalize
    flat_X = X_dev.reshape(-1, input_channels)
    x_mean = flat_X.mean(axis=0)
    x_std = flat_X.std(axis=0) + 1e-8
    
    os.makedirs(os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR), exist_ok=True)
    norm_path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{SAVE_PREFIX}_normalization.npz")
    np.savez(norm_path, x_mean=x_mean, x_std=x_std, clip_min=clip_min, clip_max=clip_max, use_deltas=USE_DELTAS)
    
    X_dev_norm = (X_dev - x_mean) / x_std
    X_pseudo_norm = (X_pseudo - x_mean) / x_std
    
    # Reconstruct sequence mapping for correct splitting
    # For simplicity/speed, we will split dev_ids and map using a simple mask
    # (Assuming X_dev is ordered by sequence, which it is, but we lost the boundaries.
    #  However, we can just split by index if we are careful, or re-build seq map.
    #  Since we loaded df_dev sorted by seq_ix/step, the samples are blocked by seq_ix.
    #  We just need to know how many samples per sequence.)
    
    # Or just map each sample back to seq_ix.
    # Efficient way:
    samples_per_seq = df_dev.groupby("seq_ix")["need_prediction"].apply(lambda x: x[N_LAGS-1:].sum()).to_dict()
    # Note: Our build function filters by need_prediction AND idx >= N_LAGS-1.
    
    # Actually, let's just re-use KFold on indices but keep sequences intact?
    # No, leakage.
    # Let's map indices to sequences.
    seq_map = []
    # Since df_dev is sorted by seq_ix, step_in_seq
    # And we iterated groupby("seq_ix") in default sort order?
    # Wait, groupby("seq_ix") sorts by key by default.
    sorted_seqs = sorted(dev_ids)
    # We iterated: for seq_ix, df_seq in df.groupby("seq_ix"):
    
    # So X_dev is ordered by sorted_seqs.
    current_idx = 0
    for seq_ix in sorted_seqs:
        count = samples_per_seq.get(seq_ix, 0) # might be slightly off if we missed some due to boundary conditions?
        # Let's trust the count from our manual simulation:
        # The build loop logic:
        # for idx in range(T-1): if need_pred[idx] and idx >= 9
        # We can recalc this exactly.
        pass 
    
    # Recalculate counts exactly
    seq_counts = []
    for seq_ix in sorted_seqs:
        # We need the original DF subset
        sub = df_dev[df_dev.seq_ix == seq_ix]
        # Logic: range(999), if need_pred[i] and i>=9
        # need_pred is 1 for 100..998.
        # So indices 100..998 are all valid.
        # Count = 999 - 100 = 899 samples per sequence.
        # Is this constant? Yes, per report.
        seq_counts.append(899)
        
    # Build seq_map
    seq_map = np.repeat(sorted_seqs, seq_counts)
    
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_results = []
    
    for fold_i, (train_idx_seq, val_idx_seq) in enumerate(kf.split(sorted_seqs)):
        train_seqs = np.array(sorted_seqs)[train_idx_seq]
        val_seqs = np.array(sorted_seqs)[val_idx_seq]
        
        train_mask = np.isin(seq_map, train_seqs)
        val_mask = np.isin(seq_map, val_seqs)
        
        X_tr_fold = X_dev_norm[train_mask]
        y_tr_fold = y_dev[train_mask]
        X_val_fold = X_dev_norm[val_mask]
        y_val_fold = y_dev[val_mask]
        
        best_state, best_r2 = train_one_fold(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, fold_i)
        fold_results.append(best_r2)
        
        save_path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{SAVE_PREFIX}_fold{fold_i}.pth")
        torch.save({
            "state_dict": best_state,
            "config": {
                "use_deltas": USE_DELTAS, 
                "num_blocks": NUM_BLOCKS, 
                "ff_dim": FF_DIM, 
                "dropout": DROPOUT
            },
            "n_lags": N_LAGS
        }, save_path)
        
    print(f"\nMean CV R2: {np.mean(fold_results):.5f}")
    
    # Pseudo-LB Eval
    print("Evaluating on Pseudo-LB...")
    models = []
    device = torch.device("cpu")
    for fold_i in range(CV_FOLDS):
        path = os.path.join(CURRENT_DIR, "..", WEIGHTS_DIR, f"{SAVE_PREFIX}_fold{fold_i}.pth")
        ckpt = torch.load(path)
        m = TSMixer(
            sequence_length=N_LAGS,
            prediction_length=1,
            input_channels=input_channels,
            output_channels=32,
            num_blocks=NUM_BLOCKS,
            ff_dim=FF_DIM,
            dropout_rate=DROPOUT,
            activation_fn="relu",
            normalize_before=True,
            norm_type="batch"
        )
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
        
    X_pseudo_tens = torch.from_numpy(X_pseudo_norm)
    preds_accum = np.zeros_like(y_pseudo)
    
    with torch.no_grad():
        # Batch for pseudo LB to save ram
        for i in range(0, len(X_pseudo_tens), BATCH_SIZE):
            batch = X_pseudo_tens[i:i+BATCH_SIZE]
            batch_preds = np.zeros((len(batch), 32))
            for m in models:
                out = m(batch).squeeze(1).numpy()
                batch_preds += out
            preds_accum[i:i+BATCH_SIZE] = batch_preds
            
    preds_ensemble = preds_accum / CV_FOLDS
    pseudo_r2s = [r2_score(y_pseudo[:, i], preds_ensemble[:, i]) for i in range(32)]
    print(f"Pseudo-LB Mean R2: {np.mean(pseudo_r2s):.5f}")

if __name__ == "__main__":
    main()
