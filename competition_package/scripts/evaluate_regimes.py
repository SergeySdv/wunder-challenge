import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch

# Adjust path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from src.features.extractor import FeatureExtractor
from scripts.optimize_vector_blend import load_ensemble, predict_ensemble, get_features_and_targets

def main():
    print("--- Evaluating Model Performance by Regime ---")
    device = torch.device("cpu")
    
    # 1. Load Cluster Mapping
    mapping_path = os.path.join(BASE_DIR, "..", "outputs", "regimes", "seq_to_cluster.json")
    with open(mapping_path, "r") as f:
        seq_to_cluster = json.load(f)
        
    # Convert keys to int
    seq_to_cluster = {int(k): v for k, v in seq_to_cluster.items()}
    
    # 2. Load Dataset (Full Train)
    dataset_path = os.path.join(BASE_DIR, "..", "datasets", "train.parquet")
    df = pd.read_parquet(dataset_path)
    
    # 3. Generate Predictions (subset for speed? No, let's do full to get robust stats per cluster)
    # Ideally we used OOF predictions, but running inference on train is "okay" to see relative fit
    # as long as we remember it's training error (optimistic).
    # HOWEVER: v19/v21 were trained on 80% of this data. The error will be biased low.
    # BUT relative performance (Model A vs Model B) should still hold.
    
    print("Generating predictions (this might take 2-3 mins)...")
    
    # v19
    models_dir = "models"
    norm_v19 = np.load(os.path.join(models_dir, "lag_mlp_normalization.npz"))
    extractor_v19 = FeatureExtractor(n_lags=10, clip_min=norm_v19["clip_min"], clip_max=norm_v19["clip_max"], use_spreads=False)
    X_v19, y_level = get_features_and_targets(df, extractor_v19, target_mode="level")
    
    models_v19 = load_ensemble("lag_mlp", use_spreads=False, device=device)
    # Predict in chunks to save RAM if needed, but 500k rows * 32 floats is small (~60MB).
    preds_v19 = predict_ensemble(models_v19, X_v19, norm_v19, device)
    
    # v21
    norm_v21 = np.load(os.path.join(models_dir, "lag_mlp_v21_normalization.npz"))
    extractor_v21 = FeatureExtractor(n_lags=10, clip_min=norm_v21["clip_min"], clip_max=norm_v21["clip_max"], use_spreads=True)
    X_v21, y_residual = get_features_and_targets(df, extractor_v21, target_mode="residual")
    
    models_v21 = load_ensemble("lag_mlp_v21", use_spreads=True, device=device)
    preds_v21_resid = predict_ensemble(models_v21, X_v21, norm_v21, device)
    
    # Reconstruct v21 level
    # Warning: get_features_and_targets iterates seq_ix. We need to map predictions back to seq_ix to assign clusters.
    # The X_list was built by iterating `df.groupby("seq_ix")`.
    # We need to reconstruct the `seq_ix` array corresponding to X rows.
    
    print("Mapping rows to clusters...")
    row_clusters = []
    N_LAGS = 10
    
    # Re-iterate to build cluster index (fast)
    for seq_ix, df_seq in df.groupby("seq_ix"):
        # logic matches get_features_and_targets
        need_pred = df_seq["need_prediction"].values
        T = len(df_seq)
        cluster_id = seq_to_cluster.get(seq_ix, -1)
        
        for idx in range(T - 1):
            if not need_pred[idx]: continue
            if idx < N_LAGS - 1: continue
            row_clusters.append(cluster_id)
            
    row_clusters = np.array(row_clusters)
    
    # Construct v21 level pred
    prev_state = y_level - y_residual
    preds_v21 = prev_state + preds_v21_resid
    
    # 4. Compute Metrics per Cluster
    print("\n--- Results by Regime (Train Data) ---")
    clusters = sorted(np.unique(row_clusters))
    
    for c in clusters:
        mask = (row_clusters == c)
        n_samples = np.sum(mask)
        if n_samples == 0: continue
        
        y_true_c = y_level[mask]
        p19_c = preds_v19[mask]
        p21_c = preds_v21[mask]
        
        # R2 per feature then mean
        r2_19 = []
        r2_21 = []
        r2_blend = []
        
        for i in range(32):
            r2_19.append(r2_score(y_true_c[:, i], p19_c[:, i]))
            r2_21.append(r2_score(y_true_c[:, i], p21_c[:, i]))
            # Simple 0.5 blend
            blend = 0.5 * p19_c[:, i] + 0.5 * p21_c[:, i]
            r2_blend.append(r2_score(y_true_c[:, i], blend))
            
        m19 = np.mean(r2_19)
        m21 = np.mean(r2_21)
        mblend = np.mean(r2_blend)
        
        print(f"Cluster {c} (N={n_samples}):")
        print(f"  v19 (Level):    {m19:.5f}")
        print(f"  v21 (Residual): {m21:.5f}")
        print(f"  Blend (0.5):    {mblend:.5f}")
        print(f"  Winner:         {'v19' if m19 > m21 else 'v21'}")
        print("-" * 30)

if __name__ == "__main__":
    main()
