import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Adjust path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from src.features.extractor import FeatureExtractor

# --- Config ---
BATCH_SIZE = 1024
EPOCHS = 10
LR = 1e-3
HIDDEN_DIM = 64

class RegimeClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, n_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    print("--- Training Regime Classifier ---")
    
    # 1. Load Data & Mappings
    mapping_path = os.path.join(BASE_DIR, "..", "outputs", "regimes", "seq_to_cluster.json")
    with open(mapping_path, "r") as f:
        seq_to_cluster = json.load(f)
    seq_to_cluster = {int(k): v for k, v in seq_to_cluster.items()}
    
    dataset_path = os.path.join(BASE_DIR, "..", "datasets", "train.parquet")
    df = pd.read_parquet(dataset_path)
    
    # Load normalization to scale inputs consistently
    # Using v19 norm as base
    norm_path = os.path.join(BASE_DIR, "..", "models", "lag_mlp_normalization.npz")
    norm = np.load(norm_path)
    x_mean = norm["x_mean"]
    x_std = norm["x_std"]
    
    # 2. Build Dataset
    # We only need a subset of windows to train this quickly
    # Let's sample 20% of the data
    print("Building dataset (20% subset)...")
    
    feature_cols = [str(i) for i in range(32)]
    extractor = FeatureExtractor(n_lags=10, clip_min=norm["clip_min"], clip_max=norm["clip_max"])
    
    X_list = []
    y_list = []
    
    # Group by seq
    # Sample sequences first to avoid leakage? 
    # No, we want to predict regime from window. Regime is sequence-level property.
    # So we must split by sequence.
    all_seqs = df["seq_ix"].unique()
    train_seqs, val_seqs = train_test_split(all_seqs, test_size=0.2, random_state=42)
    
    # Helper to process a list of sequences
    def process_seqs(seq_ids, desc):
        Xs, ys = [], []
        sub_df = df[df["seq_ix"].isin(seq_ids)]
        for seq_ix, grp in sub_df.groupby("seq_ix"):
            cluster = seq_to_cluster[seq_ix]
            states = grp[feature_cols].values
            steps = grp["step_in_seq"].values
            
            # Just take every 10th window to save memory/time
            # This is enough to learn the mapping "window -> regime"
            indices = range(9, len(grp), 10) 
            
            for idx in indices:
                raw_slice = states[idx-9 : idx+1]
                feat = extractor.build_window_features(raw_slice, steps[idx])
                Xs.append(feat)
                ys.append(cluster)
        return np.stack(Xs), np.array(ys)

    X_train, y_train = process_seqs(train_seqs, "Train")
    X_val, y_val = process_seqs(val_seqs, "Val")
    
    # Normalize
    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # 3. Train
    device = torch.device("cpu")
    n_classes = len(np.unique(y_train))
    model = RegimeClassifier(X_train.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long())
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Val Acc
        model.eval()
        val_preds = []
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val))
            val_preds = torch.argmax(val_logits, dim=1).numpy()
            
        acc = accuracy_score(y_val, val_preds)
        print(f"Epoch {epoch+1}: Loss {epoch_loss/len(train_dl):.4f}, Val Acc {acc:.4f}")
        
    print("\nClassification Report (Val):")
    print(classification_report(y_val, val_preds))
    
    # Save Model
    save_path = os.path.join(BASE_DIR, "..", "models", "regime_classifier.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved regime classifier to {save_path}")

if __name__ == "__main__":
    main()
