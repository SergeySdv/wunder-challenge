import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# --- Config ---
SEED = 42
BATCH_SIZE = 1024
HIDDEN_DIM = 128
EPOCHS = 10
LR = 1e-3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    return df

class Classifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM // 2, 1)
        )
        
    def forward(self, x):
        return torch.sigmoid(self.net(x))

def compute_sequence_stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute summary statistics for each sequence to use as features for the adversary.
    We use: Mean, Std, Min, Max of the first 10 features (raw data).
    """
    seq_stats = []
    seq_labels = []
    
    # Identify current Pseudo-LB split (random 10%)
    all_seqs = df["seq_ix"].unique()
    rng = np.random.default_rng(999) # PSEUDO_LB_SEED used in other scripts
    rng.shuffle(all_seqs)
    n_pseudo = int(len(all_seqs) * 0.10)
    pseudo_ids = set(all_seqs[:n_pseudo])
    
    # Feature columns (using first 10 raw features for simplicity/speed)
    cols = [str(i) for i in range(10)] 
    
    print(f"Computing sequence stats for {len(all_seqs)} sequences...")
    for seq_ix, group in tqdm(df.groupby("seq_ix")):
        data = group[cols].values
        
        # Compute aggregate stats
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        
        # Concat
        features = np.concatenate([means, stds, mins, maxs])
        seq_stats.append(features)
        
        # Label: 1 if Pseudo-LB, 0 if Train
        label = 1.0 if seq_ix in pseudo_ids else 0.0
        seq_labels.append(label)
        
    return np.stack(seq_stats).astype(np.float32), np.array(seq_labels).astype(np.float32)

def train_adversary(X, y):
    device = torch.device("cpu")
    
    # 5-Fold CV to get robust AUC
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    
    print("\nTraining Adversarial Classifier (Train vs Pseudo-LB)...")
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        model = Classifier(X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss()
        
        X_tr, y_tr = torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx])
        X_val, y_val = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])
        
        ds_tr = TensorDataset(X_tr, y_tr)
        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
        
        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in dl_tr:
                optimizer.zero_grad()
                loss = criterion(model(xb).squeeze(), yb)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            preds = model(X_val).squeeze().cpu().numpy()
            
        auc = roc_auc_score(y[val_idx], preds)
        aucs.append(auc)
        print(f"Fold {fold}: AUC = {auc:.4f}")
        
    mean_auc = np.mean(aucs)
    print(f"\nMean AUC: {mean_auc:.4f}")
    return mean_auc

def main():
    dataset_path = os.path.join(BASE_DIR, "..", "datasets", "train.parquet")
    if not os.path.exists(dataset_path):
        print("Dataset not found.")
        return
        
    df = load_dataset(dataset_path)
    
    # 1. Build Sequence-Level Features
    X, y = compute_sequence_stats(df)
    
    # 2. Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # 3. Train Adversary
    auc = train_adversary(X, y)
    
    print("\n--- Interpretation ---")
    if auc > 0.70:
        print("CRITICAL: Strong shift detected. Pseudo-LB is significantly different from Train.")
    elif auc > 0.60:
        print("WARNING: Moderate shift. Pseudo-LB has distinct characteristics.")
    else:
        print("PASS: No significant shift detected (AUC ~ 0.5). Random split is valid.")

if __name__ == "__main__":
    main()
