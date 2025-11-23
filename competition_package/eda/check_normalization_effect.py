import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_model import load_dataset, compute_winsorization_bounds

def check_normalization_and_filtering():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")
    
    print(f"Loading data from {dataset_path}...")
    df = load_dataset(dataset_path)
    
    # Use dev set logic to be consistent with training
    all_seqs = df["seq_ix"].unique()
    rng = np.random.default_rng(999) # Matches PSEUDO_LB_SEED
    rng.shuffle(all_seqs)
    n_pseudo = int(len(all_seqs) * 0.10)
    dev_ids = all_seqs[n_pseudo:]
    df_dev = df[df["seq_ix"].isin(dev_ids)].copy()
    
    print("Computing Winsorization bounds (0.1% - 99.9%)...")
    clip_min, clip_max = compute_winsorization_bounds(df_dev, 0.001, 0.999)
    
    feature_cols = [str(i) for i in range(32)]
    
    # Select a sequence with high variance to visualize effect
    # Feature 3 had high R2, often implies trending/volatile
    target_feat = "3"
    feat_idx = 3
    
    # Find a sequence with a "surge" (value outside clip bounds)
    # Check Feature 3
    raw_vals = df_dev[target_feat].values
    lower = clip_min[feat_idx]
    upper = clip_max[feat_idx]
    
    outliers = np.where((raw_vals < lower) | (raw_vals > upper))[0]
    
    if len(outliers) > 0:
        print(f"Found {len(outliers)} outliers in Feature {target_feat} (outside [{lower:.4f}, {upper:.4f}])")
        # Pick a sequence containing an outlier
        # Map row index back to sequence... easier to just iterate
        sample_seq = None
        for seq_ix in df_dev["seq_ix"].unique():
            seq_data = df_dev[df_dev["seq_ix"] == seq_ix][target_feat].values
            if np.any((seq_data < lower) | (seq_data > upper)):
                sample_seq = seq_ix
                break
        
        if sample_seq is None:
            sample_seq = dev_ids[0]
    else:
        print("No outliers found in Feature 3 (rare!). Using first sequence.")
        sample_seq = dev_ids[0]
        
    print(f"Visualizing Sequence {sample_seq} for Feature {target_feat}...")
    
    seq_df = df_dev[df_dev["seq_ix"] == sample_seq].sort_values("step_in_seq")
    raw_seq = seq_df[target_feat].values
    
    # Apply Transformations
    # 1. Winsorization
    clipped_seq = np.clip(raw_seq, lower, upper)
    
    # 2. Normalization
    # Note: We compute mean/std on the CLIPPED dev set, as per train_model.py logic?
    # In train_model.py:
    #   X_dev, y_dev = build_supervised_dataset(df_dev, clip_min, clip_max) -> Applies clip
    #   x_mean = X_dev.mean() -> Mean of clipped data
    
    # Let's approximate global mean/std from this feature in df_dev after clipping
    all_feat_vals = df_dev[target_feat].values
    all_feat_clipped = np.clip(all_feat_vals, lower, upper)
    
    mean_val = all_feat_clipped.mean()
    std_val = all_feat_clipped.std()
    
    norm_seq = (clipped_seq - mean_val) / std_val
    
    print(f"Global Stats for F{target_feat} (Clipped): Mean={mean_val:.4f}, Std={std_val:.4f}")
    print(f"Clip Range: [{lower:.4f}, {upper:.4f}]")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 1. Raw
    axes[0].plot(raw_seq, color='black', label='Raw')
    axes[0].axhline(lower, color='red', linestyle='--', label='Lower Clip')
    axes[0].axhline(upper, color='red', linestyle='--', label='Upper Clip')
    axes[0].set_title(f"Raw Data (Seq {sample_seq}, Feat {target_feat})")
    axes[0].legend()
    
    # 2. Clipped
    axes[1].plot(clipped_seq, color='blue', label='Winsorized')
    axes[1].set_title(f"Winsorized (Surge Filtered)")
    axes[1].legend()
    
    # 3. Normalized
    axes[2].plot(norm_seq, color='green', label='Normalized')
    axes[2].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[2].axhline(1, color='gray', linestyle=':', alpha=0.5)
    axes[2].axhline(-1, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_title(f"Normalized ((x - mu) / sigma)")
    axes[2].set_ylabel("Std Devs")
    axes[2].legend()
    
    plt.tight_layout()
    out_path = os.path.join(base_dir, "check_norm_surge.png")
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")
    
    # Statistics Check
    print("\n--- Statistics Check (Feature 3 Global) ---")
    print(f"Raw Range:     [{all_feat_vals.min():.4f}, {all_feat_vals.max():.4f}]")
    print(f"Clipped Range: [{all_feat_clipped.min():.4f}, {all_feat_clipped.max():.4f}]")
    
    norm_all = (all_feat_clipped - mean_val) / std_val
    print(f"Norm Mean: {norm_all.mean():.6f} (Should be ~0)")
    print(f"Norm Std:  {norm_all.std():.6f} (Should be ~1)")

if __name__ == "__main__":
    check_normalization_and_filtering()
