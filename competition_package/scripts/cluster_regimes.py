import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- Config ---
N_CLUSTERS = 5
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "regimes")

def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    return df

def compute_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one vector of meta-features per sequence.
    Meta-features:
    - Volatility (std of features)
    - Trend Strength (abs correlation with time)
    - Mean Level (mean of features)
    - Auto-correlation (lag-1 corr)
    """
    print("Computing meta-features per sequence...")
    
    # We'll use a subset of representative features to save time/noise.
    # Feature 0 is often a good proxy. Also mean across all features.
    feature_cols = [str(i) for i in range(32)]
    
    meta_list = []
    seq_ids = []
    
    for seq_ix, grp in df.groupby("seq_ix"):
        data = grp[feature_cols].values # (1000, 32)
        
        # 1. Global Volatility (mean std across dims)
        volatility = np.mean(np.std(data, axis=0))
        
        # 2. Global Mean (level)
        level = np.mean(np.mean(data, axis=0))
        
        # 3. Trendiness (Avg absolute correlation with time step)
        # A simple proxy is linear regression slope magnitude averaged
        time_steps = np.arange(len(data))
        corrs = []
        for i in range(32):
            # Fast correlation
            c = np.corrcoef(time_steps, data[:, i])[0, 1]
            corrs.append(abs(c))
        trend_strength = np.mean(corrs)
        
        # 4. Roughness / Noise (Avg Lag-1 Autocorrelation)
        acs = []
        for i in range(32):
            s = data[:, i]
            if np.std(s) < 1e-9:
                acs.append(0)
            else:
                ac = np.corrcoef(s[:-1], s[1:])[0, 1]
                acs.append(ac)
        autocorr = np.mean(acs)
        
        # 5. Tail Risk (Kurtosis proxy: max abs deviation from mean / std)
        # aggregated
        z_scores = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
        tail_risk = np.max(np.abs(z_scores))
        
        meta_list.append([volatility, level, trend_strength, autocorr, tail_risk])
        seq_ids.append(seq_ix)
        
    meta_df = pd.DataFrame(
        meta_list, 
        columns=["Volatility", "Level", "Trend", "Autocorr", "TailRisk"],
        index=seq_ids
    )
    return meta_df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dataset_path = os.path.join(BASE_DIR, "..", "datasets", "train.parquet")
    df = load_dataset(dataset_path)
    
    # 1. Compute Meta Features
    meta_df = compute_meta_features(df)
    
    # 2. Normalize
    scaler = StandardScaler()
    meta_scaled = scaler.fit_transform(meta_df)
    
    # 3. Clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(meta_scaled)
    
    meta_df["Cluster"] = clusters
    
    # 4. Analysis
    print("\n--- Cluster Profiles ---")
    print(meta_df.groupby("Cluster").mean())
    
    # Count per cluster
    counts = meta_df["Cluster"].value_counts().sort_index()
    print("\n--- Cluster Counts ---")
    print(counts)
    
    # 5. Save Mapping
    mapping = meta_df["Cluster"].to_dict()
    mapping_path = os.path.join(OUTPUT_DIR, "seq_to_cluster.json")
    # Convert keys to str for JSON
    mapping_str = {str(k): int(v) for k, v in mapping.items()}
    with open(mapping_path, "w") as f:
        json.dump(mapping_str, f, indent=2)
    print(f"\nSaved sequence mapping to {mapping_path}")
    
    # 6. Visualization (PCA)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(meta_scaled)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=coords[:,0], y=coords[:,1], hue=clusters, palette="viridis", s=60)
    plt.title(f"Sequence Regimes (PCA Projection) - {N_CLUSTERS} Clusters")
    plt.xlabel("PC1 (Variance?)")
    plt.ylabel("PC2 (Trend?)")
    
    plot_path = os.path.join(OUTPUT_DIR, "regime_clusters.png")
    plt.savefig(plot_path)
    print(f"Saved visualization to {plot_path}")

if __name__ == "__main__":
    main()
