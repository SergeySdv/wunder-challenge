import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import argparse
from tqdm import tqdm

# Adjust path to allow imports from src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from src.features.extractor import FeatureExtractor, feature_dim

# --- Configuration ---
SEQ_LEN = 1000
FEATURE_DIM = 1187 # v19 features (10 lags, spreads=True)
HIDDEN_DIM = 256
LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 20
LR = 5e-4
SEED = 42

# --- Data Loading & Preprocessing ---

def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
    return df

def compute_winsorization_bounds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = [str(i) for i in range(32)]
    data = df[feature_cols].values
    lower = np.quantile(data, 0.001, axis=0).astype(np.float32)
    upper = np.quantile(data, 0.999, axis=0).astype(np.float32)
    return lower, upper

def precompute_sequence_data(
    df: pd.DataFrame, 
    clip_min: np.ndarray, 
    clip_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts the DataFrame into 3D tensors for RNN training.
    
    Returns:
        X: (N_seq, T, Feature_Dim) - Inputs
        y: (N_seq, T, 32) - Targets (Residuals)
        mask: (N_seq, T) - Boolean mask (True where prediction is needed)
    """
    seq_ids = df["seq_ix"].unique()
    n_seqs = len(seq_ids)
    
    # Pre-allocate arrays
    # We assume standard v19 features with spreads -> dim 1187
    dim = feature_dim(n_lags=10, use_spreads=True)
    
    X_all = np.zeros((n_seqs, SEQ_LEN, dim), dtype=np.float32)
    y_all = np.zeros((n_seqs, SEQ_LEN, 32), dtype=np.float32)
    mask_all = np.zeros((n_seqs, SEQ_LEN), dtype=bool)
    
    print(f"Pre-computing features for {n_seqs} sequences...")
    
    feature_cols = [str(i) for i in range(32)]
    
    # Group by sequence for faster processing
    # Note: df is already sorted
    grouped = df.groupby("seq_ix")
    
    # Initialize extractor template
    # We'll reset it for each sequence
    extractor = FeatureExtractor(n_lags=10, clip_min=clip_min, clip_max=clip_max, use_spreads=True)
    
    for i, (seq_ix, df_seq) in enumerate(tqdm(grouped, total=n_seqs)):
        extractor.reset()
        
        states = df_seq[feature_cols].values.astype(np.float32)
        steps = df_seq["step_in_seq"].values
        needs = df_seq["need_prediction"].values.astype(bool)
        
        # Process each step
        # RNN needs input at t to predict t+1
        # X[t] contains features known at time t
        # y[t] contains target at time t+1 (residual: state[t+1] - state[t])
        # mask[t] is True if we need to predict t+1
        
        seq_features = []
        
        for t in range(len(df_seq)):
            current_state = states[t]
            step = steps[t]
            
            # Stream returns features if buffer is full, else None
            feat = extractor.stream(current_state, step, seq_ix)
            
            if feat is None:
                # Warmup phase: Feed zeros (will be normalized later, but keeps shape)
                # Alternatively, we could feed a padded version of current state
                # but zeros is standard for "unknown history".
                feat = np.zeros(dim, dtype=np.float32)
            
            X_all[i, t, :] = feat
            
            if t < len(df_seq) - 1:
                # Target is residual to next step
                target_resid = states[t+1] - current_state
                y_all[i, t, :] = target_resid
                mask_all[i, t] = needs[t] # need_prediction at t means we predict t+1
            else:
                # Last step has no target
                pass
                
    return X_all, y_all, mask_all

class SequenceDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.mask = torch.from_numpy(mask)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]

# --- Model ---

class FeatureGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        
        self.head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, h=None):
        # x: (Batch, Seq, Input_Dim)
        encoded = self.encoder(x) # (Batch, Seq, Hidden)
        out, h_new = self.gru(encoded, h)
        pred = self.head(out)
        return pred, h_new

# --- Training ---

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    steps = 0
    
    for X_b, y_b, mask_b in loader:
        X_b, y_b, mask_b = X_b.to(device), y_b.to(device), mask_b.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (full sequence)
        preds, _ = model(X_b)
        
        # Apply mask
        # Flatten for loss computation
        preds_flat = preds[mask_b]
        y_flat = y_b[mask_b]
        
        if len(preds_flat) == 0:
            continue
            
        loss = criterion(preds_flat, y_flat)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * len(preds_flat)
        steps += len(preds_flat)
        
    return total_loss / max(steps, 1)

def validate(model, loader, device):
    model.eval()
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for X_b, y_b, mask_b in loader:
            X_b, y_b, mask_b = X_b.to(device), y_b.to(device), mask_b.to(device)
            
            preds, _ = model(X_b)
            
            preds_flat = preds[mask_b]
            y_flat = y_b[mask_b]
            
            if len(preds_flat) > 0:
                preds_list.append(preds_flat.cpu().numpy())
                targets_list.append(y_flat.cpu().numpy())
    
    if not preds_list:
        return 0.0
        
    all_preds = np.concatenate(preds_list)
    all_targets = np.concatenate(targets_list)
    
    # Calc R2 per feature
    r2s = []
    for i in range(32):
        r2s.append(r2_score(all_targets[:, i], all_preds[:, i]))
        
    return np.mean(r2s)

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="models")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_dataset(os.path.join(BASE_DIR, "..", "datasets", "train.parquet"))
    
    print("Computing winsorization stats...")
    clip_min, clip_max = compute_winsorization_bounds(df)
    
    print("Pre-computing feature sequences (this may take a moment)...")
    X_all, y_all, mask_all = precompute_sequence_data(df, clip_min, clip_max)
    
    # Normalize Features
    # We compute mean/std over the valid parts of the sequences (or all? all is simpler)
    # Let's compute over all to keep it robust to warmup zeros.
    # Actually, warmup zeros might skew mean/std.
    # Better: Compute stats only on mask=True (valid prediction steps)
    print("Normalizing...")
    X_valid_flat = X_all[mask_all]
    x_mean = np.mean(X_valid_flat, axis=0)
    x_std = np.std(X_valid_flat, axis=0) + 1e-8
    
    # Apply normalization globally
    X_all = (X_all - x_mean) / x_std
    
    # Save Normalization
    norm_path = os.path.join(args.save_dir, "feature_gru_normalization.npz")
    np.savez(norm_path, x_mean=x_mean, x_std=x_std, clip_min=clip_min, clip_max=clip_max)
    print(f"Saved normalization to {norm_path}")
    
    # Stratified Split by Volatility
    # Calculate vol per sequence (using Feature 0)
    # X_all shape: (N, T, D). Feature 0 is in the raw data, but here we have engineered features.
    # We need to go back to df or just use the first feature of X if it's raw lag 0.
    # Feature 0 is lag_flat[0] which corresponds to state[t-9]... not easy.
    # Let's use the original df grouped.
    
    volatilities = df.groupby("seq_ix")["0"].std().values
    # Sort indices by volatility
    sorted_indices = np.argsort(volatilities)
    
    # Pick every 10th for validation (Stratified 10%)
    val_indices = sorted_indices[::10]
    train_indices = np.setdiff1d(sorted_indices, val_indices)
    
    print(f"Train sequences: {len(train_indices)}, Val sequences: {len(val_indices)}")
    
    train_ds = SequenceDataset(X_all[train_indices], y_all[train_indices], mask_all[train_indices])
    val_ds = SequenceDataset(X_all[val_indices], y_all[val_indices], mask_all[val_indices])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Setup
    device = torch.device("cpu") # CPU is fine for this size, usually
    model = FeatureGRU(
        input_dim=X_all.shape[-1], 
        hidden_dim=HIDDEN_DIM, 
        output_dim=32, 
        layers=LAYERS, 
        dropout=DROPOUT
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    print("Starting training...")
    best_val_r2 = -np.inf
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_r2 = validate(model, val_loader, device)
        
        scheduler.step(val_r2)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.5f} | Val R2: {val_r2:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), os.path.join(args.save_dir, "feature_gru_best.pth"))
            
    print(f"Training Complete. Best Val R2: {best_val_r2:.5f}")

if __name__ == "__main__":
    main()
