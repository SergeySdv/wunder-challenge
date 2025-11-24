import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import r2_score
from scipy.optimize import minimize_scalar

# Adjust path to allow imports from src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from src.features.extractor import FeatureExtractor, feature_dim

# --- Configuration ---
PSEUDO_LB_SEED = 999 
MODELS_DIR = "models"
OUTPUT_FILE = os.path.join(MODELS_DIR, "alpha_blend_v22.npy")

# --- Model Definitions (Simplified for Inference) ---
class LagMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

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

def get_features_and_targets(
    df: pd.DataFrame, 
    extractor: FeatureExtractor, 
    target_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build features and targets for a given dataframe.
    target_mode: 'level' or 'residual'
    """
    feature_cols = [str(i) for i in range(32)]
    X_list = []
    y_list = []
    
    # Assuming n_lags=10 for both models as per training
    N_LAGS = 10 
    
    for seq_ix, df_seq in df.groupby("seq_ix"):
        df_seq = df_seq.sort_values("step_in_seq")
        states = df_seq[feature_cols].values
        steps = df_seq["step_in_seq"].values
        need_pred = df_seq["need_prediction"].values
        
        T = len(df_seq)
        for idx in range(T - 1):
            if not need_pred[idx]:
                continue
            if idx < N_LAGS - 1:
                continue
            
            raw_slice = states[idx - N_LAGS + 1 : idx + 1]
            features = extractor.build_window_features(raw_slice, steps[idx])
            
            target_level = states[idx+1].astype(np.float32)
            
            if target_mode == "residual":
                target = target_level - raw_slice[-1]
            else:
                target = target_level
                
            X_list.append(features)
            y_list.append(target)
            
    return np.vstack(X_list), np.vstack(y_list)

def load_ensemble(prefix: str, use_spreads: bool, device: torch.device) -> list[LagMLP]:
    models = []
    fold_files = sorted([f for f in os.listdir(MODELS_DIR) if f.startswith(f"{prefix}_fold") and f.endswith(".pth")])
    
    if not fold_files:
        raise FileNotFoundError(f"No models found for prefix {prefix}")

    # Load arch from first model
    ckpt = torch.load(os.path.join(MODELS_DIR, fold_files[0]), map_location=device)
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    output_dim = ckpt["output_dim"]
    
    for fname in fold_files:
        path = os.path.join(MODELS_DIR, fname)
        ckpt = torch.load(path, map_location=device)
        model = LagMLP(input_dim, hidden_dim, output_dim) # Dropout doesn't matter for eval
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model.to(device)
        models.append(model)
        
    return models

def predict_ensemble(models: list[LagMLP], X: np.ndarray, norm_data: dict, device: torch.device) -> np.ndarray:
    # Normalize
    X_norm = (X - norm_data["x_mean"]) / norm_data["x_std"]
    X_tensor = torch.from_numpy(X_norm).to(device)
    
    preds_accum = np.zeros((X.shape[0], 32), dtype=np.float32)
    with torch.no_grad():
        for model in models:
            preds_accum += model(X_tensor).cpu().numpy()
            
    return preds_accum / len(models)

def main():
    print("--- optimizing Vector Blend (v19 Level + v21 Residual) ---")
    device = torch.device("cpu") # CPU is enough for inference
    
    # 1. Load Pseudo-LB Data
    root_dir = os.path.abspath(os.path.join(BASE_DIR, ".."))
    dataset_path = os.path.join(root_dir, "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    all_seqs = df["seq_ix"].unique()
    
    rng = np.random.default_rng(PSEUDO_LB_SEED)
    rng.shuffle(all_seqs)
    n_pseudo = int(len(all_seqs) * 0.10)
    pseudo_lb_ids = all_seqs[:n_pseudo]
    
    df_pseudo = df[df["seq_ix"].isin(pseudo_lb_ids)].copy()
    
    # Compute winsorization bounds (from dev set logic, but here we just need to be consistent with training)
    # Ideally we load it from normalization file, but let's recompute quickly on pseudo for the extractor init
    # or better, use the one from the model being used. 
    # The extractor needs clip_min/max.
    
    # 2. Predictions from v19 (Level)
    print("Generating v19 Level Predictions...")
    norm_v19 = np.load(os.path.join(MODELS_DIR, "lag_mlp_normalization.npz"))
    extractor_v19 = FeatureExtractor(n_lags=10, clip_min=norm_v19["clip_min"], clip_max=norm_v19["clip_max"], use_spreads=False)
    X_v19, y_level = get_features_and_targets(df_pseudo, extractor_v19, target_mode="level")
    
    models_v19 = load_ensemble("lag_mlp", use_spreads=False, device=device)
    preds_v19 = predict_ensemble(models_v19, X_v19, norm_v19, device)
    
    # 3. Predictions from v21 (Residual)
    print("Generating v21 Residual Predictions...")
    norm_v21 = np.load(os.path.join(MODELS_DIR, "lag_mlp_v21_normalization.npz"))
    extractor_v21 = FeatureExtractor(n_lags=10, clip_min=norm_v21["clip_min"], clip_max=norm_v21["clip_max"], use_spreads=True)
    X_v21, y_residual = get_features_and_targets(df_pseudo, extractor_v21, target_mode="residual")
    
    models_v21 = load_ensemble("lag_mlp_v21", use_spreads=True, device=device)
    preds_v21_resid = predict_ensemble(models_v21, X_v21, norm_v21, device)
    
    # Reconstruct level from residual predictions
    # Note: y_residual = y_level - prev_state. So prev_state = y_level - y_residual.
    # But better to get prev_state directly or infer it.
    # y_level and y_residual are aligned row-by-row.
    prev_state = y_level - y_residual
    preds_v21 = prev_state + preds_v21_resid
    
    # 4. Optimize Alpha per Feature
    print("Optimizing Alphas...")
    alpha_vector = np.zeros(32, dtype=np.float32)
    scores_v19 = []
    scores_v21 = []
    scores_blend = []
    
    for i in range(32):
        y_true = y_level[:, i]
        p1 = preds_v19[:, i]
        p2 = preds_v21[:, i]
        
        r2_v19 = r2_score(y_true, p1)
        r2_v21 = r2_score(y_true, p2)
        scores_v19.append(r2_v19)
        scores_v21.append(r2_v21)
        
        def objective(a):
            # maximize R2 -> minimize -R2 (or MSE)
            # Using MSE is more stable for optimization
            blend = a * p1 + (1 - a) * p2
            return np.mean((y_true - blend)**2)
            
        res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        alpha_vector[i] = res.x
        
        # Calc blend score
        blend_final = res.x * p1 + (1 - res.x) * p2
        scores_blend.append(r2_score(y_true, blend_final))
        
        print(f"Feat {i:02d}: v19={r2_v19:.4f}, v21={r2_v21:.4f} -> Alpha={res.x:.2f}, Blend={scores_blend[-1]:.4f}")
        
    mean_v19 = np.mean(scores_v19)
    mean_v21 = np.mean(scores_v21)
    mean_blend = np.mean(scores_blend)
    
    print(f"\nResults Summary on Pseudo-LB:")
    print(f"v19 (Level) Mean R2: {mean_v19:.5f}")
    print(f"v21 (Resid) Mean R2: {mean_v21:.5f}")
    print(f"v22 (Blend) Mean R2: {mean_blend:.5f}")
    print(f"Improvement: +{mean_blend - max(mean_v19, mean_v21):.5f}")
    
    # Save Alpha Vector
    np.save(OUTPUT_FILE, alpha_vector)
    print(f"Saved optimized alpha vector to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
