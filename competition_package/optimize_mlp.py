import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import r2_score
from train_model import load_dataset, build_supervised_dataset, compute_winsorization_bounds

# --- Configuration ---
N_TRIALS = 20  # Number of random search trials
EPOCHS_PER_TRIAL = 10 # Reduce epochs for speed during search
VAL_FRACTION = 0.2 # Single validation split
SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model (Configurable) ---
class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim, h1, h2, dropout, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def train_and_evaluate(params, X_train, y_train, X_val, y_val):
    device = torch.device("cpu")
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = ConfigurableMLP(
        input_dim, 
        params["h1"], 
        params["h2"], 
        params["dropout"], 
        output_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()
    
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)
    
    best_r2 = -1e9
    
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb).numpy())
                targets.append(yb.numpy())
        
        preds = np.vstack(preds)
        targets = np.vstack(targets)
        
        # Mean R2
        r2s = [r2_score(targets[:, i], preds[:, i]) for i in range(targets.shape[1])]
        mean_r2 = np.mean(r2s)
        
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            
    return best_r2

def run_optimization():
    print(f"Loading data...")
    dataset_path = os.path.join(BASE_DIR, "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    
    # Subsample for speed if needed, or use full dev set (minus Pseudo-LB)
    # To be comparable with v10, we should use the Dev Set logic.
    # But for optimization speed, let's just use a fixed 20% holdout on the Dev Set.
    
    all_seqs = df["seq_ix"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_seqs)
    
    # Hold out Pseudo-LB (10%) just to be safe/consistent, even if we don't use it here
    n_pseudo = int(len(all_seqs) * 0.10)
    dev_ids = all_seqs[n_pseudo:]
    
    # Now split Dev into Train/Val for Optimization
    n_val = int(len(dev_ids) * VAL_FRACTION)
    val_ids = dev_ids[:n_val]
    train_ids = dev_ids[n_val:]
    
    df_train = df[df["seq_ix"].isin(train_ids)].copy()
    df_val = df[df["seq_ix"].isin(val_ids)].copy()
    
    print(f"Optimization Train Seqs: {len(train_ids)}, Val Seqs: {len(val_ids)}")
    
    # Compute Winsorization & Norm on Train (this subset)
    clip_min, clip_max = compute_winsorization_bounds(df_train, 0.001, 0.999)
    
    X_train, y_train = build_supervised_dataset(df_train, clip_min, clip_max)
    X_val, y_val = build_supervised_dataset(df_val, clip_min, clip_max)
    
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    
    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std
    
    print(f"Data ready. Starting {N_TRIALS} trials...")
    
    history = []
    best_score = -1e9
    best_params = {}
    
    for i in range(N_TRIALS):
        # Sample Params
        params = {
            "h1": random.choice([256, 512, 1024]),
            "h2": random.choice([128, 256, 512]),
            "dropout": random.choice([0.1, 0.2, 0.3, 0.4]),
            "lr": random.choice([1e-3, 5e-4]),
            "batch_size": random.choice([512, 1024])
        }
        
        # Constraint: h2 <= h1 usually makes sense for funnel
        if params["h2"] > params["h1"]:
            params["h2"] = params["h1"] // 2
            
        print(f"Trial {i+1}/{N_TRIALS}: {params}")
        
        score = train_and_evaluate(params, X_train, y_train, X_val, y_val)
        print(f"  -> Val R2: {score:.5f}")
        
        history.append({**params, "score": score})
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"  *** New Best! ***")
            
    print("\n=== Optimization Finished ===")
    print(f"Best Score: {best_score:.5f}")
    print(f"Best Params: {best_params}")
    
    # Save to CSV
    pd.DataFrame(history).sort_values("score", ascending=False).to_csv(
        os.path.join(BASE_DIR, "mlp_optimization_results.csv"), index=False
    )

if __name__ == "__main__":
    run_optimization()
