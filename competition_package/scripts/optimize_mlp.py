import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import optuna
from sklearn.metrics import r2_score
from train_model import load_dataset, build_supervised_dataset, compute_winsorization_bounds

# --- Configuration ---
N_TRIALS = 50  # TPE converges faster than random search
EPOCHS_PER_TRIAL = 15 
SEED = 42
VAL_FRACTION = 0.2 # Single validation split for speed (CV inside optuna is too slow)

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

def objective(trial):
    # 1. Sample Hyperparameters
    params = {
        "h1": trial.suggest_int("h1", 64, 512),
        "h2": trial.suggest_int("h2", 32, 256),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    }
    
    # Constraint: Funnel shape usually works best
    if params["h2"] > params["h1"]:
        params["h2"] = params["h1"] // 2

    # 2. Setup Training
    device = torch.device("cpu")
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = ConfigurableMLP(
        input_dim, params["h1"], params["h2"], params["dropout"], output_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=params["lr"], 
        weight_decay=params["weight_decay"]
    )
    loss_fn = nn.MSELoss()
    
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)
    
    # 3. Training Loop with Pruning
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            
        # Validation
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
        
        # Report to Optuna
        trial.report(mean_r2, epoch)
        
        # Handle Pruning (kill bad trials)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return mean_r2

if __name__ == "__main__":
    print(f"Loading data...")
    dataset_path = os.path.join(BASE_DIR, "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    
    # Data Split (Single Holdout for Speed)
    all_seqs = df["seq_ix"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_seqs)
    
    n_pseudo = int(len(all_seqs) * 0.10)
    dev_ids = all_seqs[n_pseudo:] # Use full dev set minus Pseudo-LB
    
    n_val = int(len(dev_ids) * VAL_FRACTION)
    val_ids = dev_ids[:n_val]
    train_ids = dev_ids[n_val:]
    
    df_train = df[df["seq_ix"].isin(train_ids)].copy()
    df_val = df[df["seq_ix"].isin(val_ids)].copy()
    
    print(f"Optimization Train Seqs: {len(train_ids)}, Val Seqs: {len(val_ids)}")
    
    # Preprocessing
    clip_min, clip_max = compute_winsorization_bounds(df_train, 0.001, 0.999)
    
    # Making these global so objective function can see them without passing large objects
    global X_train, y_train, X_val, y_val
    X_train, y_train = build_supervised_dataset(df_train, clip_min, clip_max)
    X_val, y_val = build_supervised_dataset(df_val, clip_min, clip_max)
    
    # Normalize
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    
    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std
    
    print(f"Data ready (Dim: {X_train.shape[1]}). Starting {N_TRIALS} TPE trials...")
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n=== Optimization Finished ===")
    print(f"Best Score: {study.best_value:.5f}")
    print(f"Best Params: {study.best_params}")
    
    # Save to CSV
    res_df = study.trials_dataframe()
    res_df.to_csv(os.path.join(BASE_DIR, "mlp_optuna_results.csv"), index=False)