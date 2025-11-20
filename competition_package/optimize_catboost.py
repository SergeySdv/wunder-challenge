import os
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from train_catboost_experiment import load_dataset, compute_winsorization_bounds, build_dataset

# --- Configuration ---
N_TRIALS = 15
SEED = 42
SUBSET_FRACTION = 0.20  # Use 20% of data for speed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def objective(trial):
    # 1. Hyperparameters to tune
    params = {
        "iterations": trial.suggest_categorical("iterations", [300, 500, 800]),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
        "depth": trial.suggest_int("depth", 4, 6),
        "colsample_bylevel": trial.suggest_categorical("colsample_bylevel", [0.05, 0.1, 0.3]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.8]),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 5),
        "bootstrap_type": "Bernoulli",
        "loss_function": "MultiRMSE",
        "verbose": 0,
        "thread_count": 4,
        "allow_writing_files": False
    }

    # 2. Load Data (Cached ideally, but for script simplicity we reload/build once)
    # We rely on global X_tr, y_tr, X_val, y_val being populated before study.optimize
    
    model = CatBoostRegressor(**params)
    
    # Train
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=20)
    
    # Evaluate
    preds = model.predict(val_pool)
    r2s = [r2_score(y_val[:, i], preds[:, i]) for i in range(32)]
    mean_r2 = np.mean(r2s)
    
    return mean_r2

if __name__ == "__main__":
    print("--- Optimizing CatBoost (Fast Mode) ---")
    
    dataset_path = os.path.join(BASE_DIR, "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    
    # Subsample Sequences
    all_seqs = df["seq_ix"].unique()
    rng = np.random.default_rng(SEED)
    subset_seqs = rng.choice(all_seqs, size=int(len(all_seqs) * SUBSET_FRACTION), replace=False)
    
    print(f"Subsampling {len(subset_seqs)} sequences ({SUBSET_FRACTION*100}%)")
    df_sub = df[df["seq_ix"].isin(subset_seqs)].copy()
    
    # Preprocessing
    print("Computing bounds & building dataset...")
    clip_min, clip_max = compute_winsorization_bounds(df_sub)
    X, y = build_dataset(df_sub, clip_min, clip_max)
    print(f"Dataset shape: {X.shape}")
    
    # Simple Train/Val Split (Last 20% of subset as Val)
    # No CV inside Optuna to save time.
    n_train = int(len(X) * 0.8)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    
    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_val, y_val)
    
    print("Starting Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n=== Best Trial ===")
    print(f"Value: {study.best_value:.5f}")
    print(f"Params: {study.best_params}")
    
    # Save
    res_df = study.trials_dataframe()
    res_df.to_csv(os.path.join(BASE_DIR, "catboost_optimization_results.csv"), index=False)
