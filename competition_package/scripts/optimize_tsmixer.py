import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import r2_score
import optuna

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, ".."))

from src.models.torchtsmixer import TSMixer

# --- Configuration ---
N_LAGS = 10
PSEUDO_LB_SEED = 999
N_EPOCHS = 15  # Fast epochs for tuning
BATCH_SIZE = 512

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
                # Delta: x[t] - x[t-1]
                # First element of delta is x[t-9] - x[t-10] (which we don't have), 
                # so we pad or diff inside window.
                # Let's do x[t] - x[t-1] for the window.
                # To keep shape (10, 32), we can do:
                # delta[i] = x[i] - x[i-1]. For i=0, define delta[0] = 0 or x[0]-x[-1] (prev sequence?)
                # Simpler: Delta vs Last Step.
                # TSMixer paper often keeps raw only.
                # Let's try: Concatenate Raw + (Raw - Raw_Mean) or Raw + (Raw - Raw[t-1])
                # Here: Let's just compute diffs along time axis.
                # Diffs: (9, 32). Pad to (10, 32) with zeros at start.
                diffs = np.zeros_like(lag_slice)
                diffs[1:] = lag_slice[1:] - lag_slice[:-1]
                
                # Stack channels: (10, 32) + (10, 32) -> (10, 64)
                combined = np.concatenate([lag_slice, diffs], axis=1)
                X_list.append(combined)
            else:
                X_list.append(lag_slice)
                
            y_list.append(states[idx+1].astype(np.float32))
            
    return np.array(X_list), np.array(y_list)

def objective(trial):
    # --- Hyperparameters ---
    use_deltas = trial.suggest_categorical("use_deltas", [True, False])
    num_blocks = trial.suggest_int("num_blocks", 1, 4)
    ff_dim = trial.suggest_int("ff_dim", 32, 256, step=32)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    
    # Data Setup (Cached globally or passed? For Optuna, re-building is safer but slow.
    # Ideally we load once outside. For simplicity, we assume global X_dev/X_pseudo pre-loaded.)
    # But 'use_deltas' changes the dataset. 
    # So we rely on global pre-built datasets for both cases.
    
    if use_deltas:
        X_tr, y_tr = DATA["train_delta"]
        X_val, y_val = DATA["val_delta"]
        input_channels = 64
    else:
        X_tr, y_tr = DATA["train_raw"]
        X_val, y_val = DATA["val_raw"]
        input_channels = 32
        
    # Model
    device = torch.device("cpu")
    model = TSMixer(
        sequence_length=N_LAGS,
        prediction_length=1,
        input_channels=input_channels,
        output_channels=32,
        num_blocks=num_blocks,
        ff_dim=ff_dim,
        dropout_rate=dropout,
        activation_fn="relu",
        normalize_before=True,
        norm_type="batch"
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Training Loop (Subset of data for speed?)
    # Let's use the full Dev set split into Train/Val (Holdout)
    # DATA["train"] is 90% of Dev, DATA["val"] is 10% of Dev (Pseudo-LB is separate).
    # Actually, let's just use Train vs Pseudo-LB to match the target metric directly.
    # Train on Dev (X_tr), Validate on Pseudo-LB (X_val).
    
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    # val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    best_r2 = -1e9
    
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(1)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        preds_accum = []
        targets_accum = []
        
        # Manual batching for validation to save memory
        with torch.no_grad():
            for i in range(0, len(X_val), BATCH_SIZE):
                xb = torch.from_numpy(X_val[i:i+BATCH_SIZE]).to(device)
                yb = y_val[i:i+BATCH_SIZE]
                out = model(xb).squeeze(1).cpu().numpy()
                preds_accum.append(out)
                targets_accum.append(yb)
                
        preds = np.vstack(preds_accum)
        targets = np.vstack(targets_accum)
        
        r2s = [r2_score(targets[:, i], preds[:, i]) for i in range(32)]
        mean_r2 = np.mean(r2s)
        
        trial.report(mean_r2, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_r2 = max(best_r2, mean_r2)
        
    return best_r2

DATA = {}

def prepare_data():
    print("Loading data...")
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
    
    print("Computing Winsorization...")
    clip_min, clip_max = compute_winsorization_bounds(df_dev)
    
    print("Building RAW datasets...")
    X_dev_raw, y_dev = build_dataset_with_deltas(df_dev, clip_min, clip_max, use_deltas=False)
    X_pseudo_raw, y_pseudo = build_dataset_with_deltas(df_pseudo, clip_min, clip_max, use_deltas=False)
    
    print("Building DELTA datasets...")
    X_dev_delta, _ = build_dataset_with_deltas(df_dev, clip_min, clip_max, use_deltas=True)
    X_pseudo_delta, _ = build_dataset_with_deltas(df_pseudo, clip_min, clip_max, use_deltas=True)
    
    # Normalize
    print("Normalizing...")
    # Raw
    raw_mean = X_dev_raw.reshape(-1, 32).mean(axis=0)
    raw_std = X_dev_raw.reshape(-1, 32).std(axis=0) + 1e-8
    DATA["train_raw"] = ((X_dev_raw - raw_mean) / raw_std, y_dev)
    DATA["val_raw"] = ((X_pseudo_raw - raw_mean) / raw_std, y_pseudo)
    
    # Delta
    delta_mean = X_dev_delta.reshape(-1, 64).mean(axis=0)
    delta_std = X_dev_delta.reshape(-1, 64).std(axis=0) + 1e-8
    DATA["train_delta"] = ((X_dev_delta - delta_mean) / delta_std, y_dev)
    DATA["val_delta"] = ((X_pseudo_delta - delta_mean) / delta_std, y_pseudo)
    
    print("Data ready.")

def main():
    prepare_data()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # 20 trials for quick check
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best params to CSV
    df = pd.DataFrame([trial.params])
    df["best_value"] = trial.value
    os.makedirs(os.path.join(CURRENT_DIR, "..", "experiments"), exist_ok=True)
    df.to_csv(os.path.join(CURRENT_DIR, "..", "experiments", "tsmixer_optuna_results.csv"), index=False)

if __name__ == "__main__":
    main()
