import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import argparse

# --- Configuration ---
N_LAGS = 10
PSEUDO_LB_SEED = 999
CV_FOLDS = 5
CV_SEED = 42
ITERATIONS = 500  # fewer trees for faster turnaround
LEARNING_RATE = 0.05
DEPTH = 6
SUBSAMPLE = 0.8  # Row subsampling for speed/robustness

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Helper Functions ---

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


def build_feature_names() -> list:
    names = []
    for k in range(10):
        for f in range(32):
            names.append(f"lag[{k}]/feat{f}")
    for k in range(10):
        for f in range(32):
            names.append(f"delta[{k}]/feat{f}")
    blocks = [
        ("mean", 32),
        ("std", 32),
        ("ac1", 32),
        ("ac2", 32),
        ("ac3", 32),
        ("acf_sum", 32),
        ("frac_above_mean", 32),
        ("q25", 32),
        ("median", 32),
        ("q75", 32),
        ("iqr", 32),
        ("skew", 32),
        ("kurt", 32),
        ("cv", 32),
        ("slope", 32),
        ("r2", 32),
        ("curvature", 32),
        ("step_val", 1),
    ]
    for name, dim in blocks:
        if dim == 1:
            names.append(name)
        else:
            for f in range(dim):
                names.append(f"{name}/feat{f}")
    return names

# --- Feature Engineering (v19 feature set: v6 streaming-safe features, no v13 kinematics) ---

def _compute_lag1_autocorr(lag_slice):
    x = lag_slice[:-1, :]
    y = lag_slice[1:, :]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    num = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    denom = np.sqrt(((x - x_mean)**2).mean(axis=0) * ((y - y_mean)**2).mean(axis=0)) + 1e-8
    return (num / denom).astype(np.float32)

def _compute_lagk_autocorr(lag_slice, lag):
    if lag >= lag_slice.shape[0]: return np.zeros(lag_slice.shape[1], dtype=np.float32)
    x = lag_slice[:-lag, :]
    y = lag_slice[lag:, :]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    num = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    denom = np.sqrt(((x - x_mean)**2).mean(axis=0) * ((y - y_mean)**2).mean(axis=0)) + 1e-8
    return (num / denom).astype(np.float32)

def _compute_frac_above_mean(lag_slice, mean_last):
    return (lag_slice > mean_last[None, :]).mean(axis=0).astype(np.float32)

def _compute_robust_window_stats(lag_slice, mean_last, std_last):
    q25 = np.percentile(lag_slice, 25, axis=0).astype(np.float32)
    median = np.percentile(lag_slice, 50, axis=0).astype(np.float32)
    q75 = np.percentile(lag_slice, 75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)
    
    denom = std_last + 1e-8
    standardized = (lag_slice - mean_last[None, :]) / denom[None, :]
    skewness = (standardized**3).mean(axis=0).astype(np.float32)
    kurtosis = ((standardized**4).mean(axis=0) - 3.0).astype(np.float32)
    cv = (std_last / (np.abs(mean_last) + 1e-8)).astype(np.float32)
    return q25, median, q75, iqr, skewness, kurtosis, cv

def _compute_trend_features(lag_slice):
    n_lags = lag_slice.shape[0]
    t = np.arange(n_lags, dtype=np.float32)
    sum_t = float(n_lags * (n_lags - 1) / 2.0)
    sum_t2 = float(n_lags * (n_lags - 1) * (2 * n_lags - 1) / 6.0)
    sum_y = lag_slice.sum(axis=0)
    sum_ty = (t[:, None] * lag_slice).sum(axis=0)
    denom = n_lags * sum_t2 - sum_t * sum_t
    
    if denom == 0.0:
        slope = np.zeros(lag_slice.shape[1], dtype=np.float32)
        r2 = np.zeros(lag_slice.shape[1], dtype=np.float32)
    else:
        slope = (n_lags * sum_ty - sum_t * sum_y) / denom
        intercept = (sum_y - slope * sum_t) / float(n_lags)
        fitted = intercept[None, :] + slope[None, :] * t[:, None]
        residual = lag_slice - fitted
        ss_res = (residual**2).sum(axis=0)
        mean_y = lag_slice.mean(axis=0)
        ss_tot = ((lag_slice - mean_y[None, :]) ** 2).sum(axis=0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        
    slope = slope.astype(np.float32)
    r2 = r2.astype(np.float32)
    
    mid = n_lags // 2
    slope_first = (lag_slice[mid] - lag_slice[0]) / float(mid)
    slope_second = (lag_slice[-1] - lag_slice[mid]) / float(n_lags - mid)
    curvature = (slope_second - slope_first).astype(np.float32)
    
    return slope, r2, curvature

def build_dataset(df, clip_min, clip_max):
    feature_cols = [str(i) for i in range(32)]
    X_list, y_list = [], []
    
    for seq_ix, df_seq in df.groupby("seq_ix"):
        df_seq = df_seq.sort_values("step_in_seq")
        states = df_seq[feature_cols].values
        steps = df_seq["step_in_seq"].values
        need_pred = df_seq["need_prediction"].values
        
        T = len(df_seq)
        for idx in range(T - 1):
            if not need_pred[idx]: continue
            if idx < N_LAGS - 1: continue
            
            raw_slice = states[idx - N_LAGS + 1 : idx + 1]
            lag_slice = np.clip(raw_slice, clip_min, clip_max).astype(np.float32)
            
            # Standard v11 Features
            lag_flat = lag_slice.reshape(-1)
            last = lag_slice[-1]
            delta_flat = (lag_slice - last).reshape(-1)
            mean_last = lag_slice.mean(axis=0).astype(np.float32)
            std_last = lag_slice.std(axis=0).astype(np.float32)
            
            ac1 = _compute_lag1_autocorr(lag_slice)
            ac2 = _compute_lagk_autocorr(lag_slice, 2)
            ac3 = _compute_lagk_autocorr(lag_slice, 3)
            acf_sum = np.abs(ac1) + np.abs(ac2) + np.abs(ac3)
            
            frac = _compute_frac_above_mean(lag_slice, mean_last)
            q25, median, q75, iqr, skew, kurt, cv = _compute_robust_window_stats(lag_slice, mean_last, std_last)
            slope, r2, curve = _compute_trend_features(lag_slice)
            
            step_val = np.array([steps[idx] / 1000.0], dtype=np.float32)
            
            features = np.concatenate([
                lag_flat, delta_flat, mean_last, std_last,
                ac1, ac2, ac3, acf_sum, frac,
                q25, median, q75, iqr, skew, kurt, cv,
                slope, r2, curve,
                step_val
            ])
            
            X_list.append(features)
            y_list.append(states[idx+1])
            
    return np.vstack(X_list), np.vstack(y_list)

def main():
    parser = argparse.ArgumentParser(description="CatBoost MultiRMSE on v19 feature set")
    parser.add_argument("--subset", type=int, default=None, help="Optional row subsample for speed (e.g., 120000).")
    parser.add_argument("--save_model_dir", type=str, default=None, help="Directory to save fold models (.cbm).")
    parser.add_argument("--save_prefix", type=str, default="catboost_v19", help="Prefix for saved models/importances.")
    parser.add_argument("--save_importance", action="store_true", help="Save feature importances CSV for first fold.")
    args = parser.parse_args()

    print("--- CatBoost (v19 feature set) ---")
    
    dataset_path = os.path.join(BASE_DIR, "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    all_seqs = df["seq_ix"].unique()
    
    # Pseudo-LB
    rng = np.random.default_rng(PSEUDO_LB_SEED)
    rng.shuffle(all_seqs)
    n_pseudo = int(len(all_seqs) * 0.10)
    pseudo_ids = all_seqs[:n_pseudo]
    dev_ids = all_seqs[n_pseudo:]
    
    df_pseudo = df[df["seq_ix"].isin(pseudo_ids)].copy()
    df_dev = df[df["seq_ix"].isin(dev_ids)].copy()
    
    print("Computing winsorization bounds...")
    clip_min, clip_max = compute_winsorization_bounds(df_dev)
    
    # 5-Fold CV
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_scores = []
    feature_names = build_feature_names()
    
    # Feature Names (approximate, for importance)
    # Base 32, Lags 10 -> 320 names
    # We won't generate all 1300 names manually, CatBoost will use indices 0..N
    
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(dev_ids)):
        print(f"\nFold {fold_i} processing...")
        train_seqs = dev_ids[train_idx]
        val_seqs = dev_ids[val_idx]
        
        # Subsample training data for speed? CatBoost on 300k rows with 1300 features is heavy.
        # Let's use 50% of training sequences to speed up the experiment.
        # rng_fold = np.random.default_rng(fold_i)
        # train_seqs_sub = rng_fold.choice(train_seqs, size=int(len(train_seqs)*0.5), replace=False)
        # df_tr = df_dev[df_dev["seq_ix"].isin(train_seqs_sub)].copy()
        # Update: Let's run FULL data. If it's too slow, we'll know.
        
        df_tr = df_dev[df_dev["seq_ix"].isin(train_seqs)].copy()
        df_val = df_dev[df_dev["seq_ix"].isin(val_seqs)].copy()
        
        print("  Building datasets...")
        X_tr, y_tr = build_dataset(df_tr, clip_min, clip_max)
        X_val, y_val = build_dataset(df_val, clip_min, clip_max)
        if args.subset is not None and args.subset < len(X_tr):
            idx = np.random.choice(len(X_tr), args.subset, replace=False)
            X_tr = X_tr[idx]
            y_tr = y_tr[idx]
        
        print(f"  Training shape: {X_tr.shape}")
        
        train_pool = Pool(X_tr, y_tr)
        val_pool = Pool(X_val, y_val)
        
        model = CatBoostRegressor(
            loss_function="MultiRMSE",
            iterations=ITERATIONS,
            learning_rate=LEARNING_RATE,
            depth=DEPTH,
            subsample=SUBSAMPLE,
            bootstrap_type="Bernoulli", # Required for subsample
            verbose=100,
            early_stopping_rounds=50,
            thread_count=4
        )
        
        model.fit(train_pool, eval_set=val_pool)
        
        # Evaluate R2
        preds = model.predict(val_pool)
        r2s = [r2_score(y_val[:, i], preds[:, i]) for i in range(32)]
        mean_r2 = np.mean(r2s)
        fold_scores.append(mean_r2)
        print(f"  Fold {fold_i} R2: {mean_r2:.5f}")
        
        # Save model per fold if requested
        if args.save_model_dir:
            os.makedirs(args.save_model_dir, exist_ok=True)
            model_path = os.path.join(args.save_model_dir, f"{args.save_prefix}_fold{fold_i}.cbm")
            model.save_model(model_path)
            print(f"  Saved fold {fold_i} model to {model_path}")

        # Save / print feature importance from first fold to avoid clutter
        if fold_i == 0:
            importances = model.get_feature_importance(train_pool)
            top_indices = np.argsort(importances)[::-1][:50]
            print("\nTop 20 Feature Indices & Scores:")
            for idx in top_indices[:20]:
                print(f"  Idx {idx}: {importances[idx]:.4f} ({feature_names[idx] if idx < len(feature_names) else 'unknown'})")
            if args.save_importance:
                os.makedirs(args.save_model_dir or BASE_DIR, exist_ok=True)
                imp_path = os.path.join(args.save_model_dir or BASE_DIR, f"{args.save_prefix}_importance.csv")
                df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
                df_imp.sort_values("importance", ascending=False).to_csv(imp_path, index=False)
                print(f"  Saved feature importances to {imp_path}")
    
    print(f"\nMean CV R2: {np.mean(fold_scores):.5f}")
    
    # Pseudo-LB using last trained model (single model proxy)
    print("Evaluating on Pseudo-LB (single model)...")
    X_pseudo, y_pseudo = build_dataset(df_pseudo, clip_min, clip_max)
    preds_pseudo = model.predict(X_pseudo)
    pseudo_r2s = [r2_score(y_pseudo[:, i], preds_pseudo[:, i]) for i in range(32)]
    print(f"Pseudo-LB R2: {np.mean(pseudo_r2s):.5f}")

if __name__ == "__main__":
    main()
