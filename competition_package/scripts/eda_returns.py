"""
Quick EDA for 32-dim return-like features:
- Spearman correlation matrix (clustered for structure)
- Per-feature distribution stats (mean/std/skew/kurt + tail quantiles)
- Hist/box plots per feature (optional)

Usage (from repo root):
  cd competition_package
  ../.venv/bin/python scripts/eda_returns.py

Outputs:
  outputs/eda/feature_stats.json
  outputs/eda/corr_spearman.csv
  outputs/eda/corr_spearman.png (clustered)
  outputs/eda/hist_feature_*.png
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    out_dir = os.path.join(root_dir, "outputs", "eda")
    ensure_dir(out_dir)

    dataset_path = os.path.join(root_dir, "datasets", "train.parquet")
    df = pd.read_parquet(dataset_path)
    feature_cols = [str(i) for i in range(32)]
    
    df_features = df[feature_cols].copy()
    
    # Winsorize at current defaults (0.1/99.9)
    data = df_features.to_numpy()
    lower = np.quantile(data, 0.001, axis=0)
    upper = np.quantile(data, 0.999, axis=0)
    data_clip = np.clip(data, lower, upper)
    df_clip = pd.DataFrame(data_clip, columns=feature_cols)

    # Spearman correlation
    print("Computing Spearman correlation...")
    corr = df_clip.corr(method="spearman")
    corr_path = os.path.join(out_dir, "corr_spearman.csv")
    corr.to_csv(corr_path)
    print(f"Saved Spearman correlation to {corr_path}")

    # Clustering to reveal structure
    print("Clustering features...")
    # Convert correlation to distance
    dist_matrix = 1 - np.abs(corr)
    # Handle floating point errors (ensure it's a valid distance matrix)
    np.fill_diagonal(dist_matrix.values, 0)
    dist_condensed = squareform(dist_matrix)
    
    # Hierarchical clustering (Ward's method)
    linkage = hierarchy.linkage(dist_condensed, method='ward')
    # Get the optimal order of leaves
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    optimal_order = dendro['leaves']
    
    # Reorder the correlation matrix
    corr_clustered = corr.iloc[optimal_order, optimal_order]
    
    # Save clustered correlation to CSV to match the plot
    corr_clustered_path = os.path.join(out_dir, "corr_spearman_clustered.csv")
    corr_clustered.to_csv(corr_clustered_path)
    print(f"Saved clustered Spearman correlation to {corr_clustered_path}")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use('dark_background')
        plt.rcParams.update({
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.facecolor": "#202020",
            "figure.facecolor": "#121212",
            "savefig.facecolor": "#121212"
        })

        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Custom colormap: Red (neg) -> Dark -> Green (pos)
        cmap = sns.diverging_palette(10, 130, as_cmap=True, s=99, l=30, center="dark")

        sns.heatmap(
            corr_clustered,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="#303030",
            cbar_kws={'shrink': 0.75, 'aspect': 30},
            ax=ax
        )
        
        # Highlight strong correlations
        for text in ax.texts:
            try:
                val = float(text.get_text())
                if abs(val) > 0.7:
                    text.set_color('yellow')
                    text.set_weight('bold')
                else:
                    text.set_color('white')
                    text.set_alpha(0.7)
            except:
                pass

        ax.set_title(
            "Correlation Matrix (Clustered)\n"
            f"Method: spearman | Variables: {len(corr.columns)} | Observations: {len(df_clip)}\n"
            f"Features reordered by similarity to reveal structure",
            color='white',
            fontsize=14
        )
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.tight_layout()
        heatmap_path = os.path.join(out_dir, "corr_spearman.png")
        plt.savefig(heatmap_path, dpi=200)
        plt.close()
        print(f"Saved clustered correlation heatmap to {heatmap_path}")
    except Exception as e:
        print(f"Skipping heatmap: {e}")

    # Per-feature stats
    stats = {}
    for col in feature_cols:
        arr = df_clip[col].to_numpy()
        stats[col] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "skew": float(pd.Series(arr).skew()),
            "kurt": float(pd.Series(arr).kurtosis()),
            "q001": float(np.quantile(arr, 0.001)),
            "q01": float(np.quantile(arr, 0.01)),
            "q99": float(np.quantile(arr, 0.99)),
            "q999": float(np.quantile(arr, 0.999)),
        }
    stats_path = os.path.join(out_dir, "feature_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved feature stats to {stats_path}")

    # Optional hist/box plots
    try:
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        
        # Only plot first 5 to save time if needed, or all
        # Plotting all as requested previously
        for col in feature_cols:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(df_clip[col], bins=50, alpha=0.7, color="steelblue")
            ax.set_title(f"Feature {col} (winsorized)", color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"hist_feature_{col}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
        print(f"Saved per-feature hists to {out_dir}")
    except Exception as e:
        print(f"Skipping per-feature hists: {e}")


if __name__ == "__main__":
    main()
