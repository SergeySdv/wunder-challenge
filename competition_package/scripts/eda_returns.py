"""
Quick EDA for 32-dim return-like features:
- Spearman correlation matrix
- Per-feature distribution stats (mean/std/skew/kurt + tail quantiles)
- Hist/box plots per feature (optional)

Usage (from repo root):
  cd competition_package
  ../.venv/bin/python scripts/eda_returns.py

Outputs:
  outputs/eda/feature_stats.json
  outputs/eda/corr_spearman.csv
  outputs/eda/corr_spearman.png (if matplotlib available)
  outputs/eda/hist_feature_*.png (optional; skip if too heavy)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    out_dir = os.path.join(root_dir, "outputs", "eda")
    ensure_dir(out_dir)

    dataset_path = os.path.join(root_dir, "datasets", "train.parquet")
    df = pd.read_parquet(dataset_path)
    # Assuming feature columns are '0' through '31'
    feature_cols = [str(i) for i in range(32)]
    
    # Identify relevant columns for correlation matrix
    # Based on the image, there are additional columns like 'MASSI_9_25', 'BuPo', etc.
    # and also 'returns_lagX', 'day_of_week_cos', 'day_of_week_sin', 'target', 'ticker'
    # For now, let's include the 32 features and a few from EDA.py and the image
    
    # For a precise match with the image, we would need to know the exact columns used
    # in the original EDA that generated the image.
    # Let's try to infer from the image and current common feature names.
    # The image shows 'returns_lagX', 'MASSI_9_25', 'BuPo', 'BePo', '-DI', '+DI', 'ADX',
    # 'ATRr_14', 'BBP', 'BBB', 'MACD_Signal', 'MACD', 'RSI_14', 'returns',
    # 'day_of_week_cos', 'day_of_week_sin', 'target', 'ticker'.
    # These are likely generated features or metadata.
    # For this script, we'll focus on the '0'-'31' features and add some mock ones for illustration
    # as the current 'train.parquet' only contains 0-31 and seq_ix, step_in_seq, need_prediction.
    # To truly replicate, we'd need to re-generate these features first.
    
    # For the purpose of making a similar plot, let's create a dummy DataFrame
    # that includes the feature_cols and some of the additional columns seen in the image.
    # This is a placeholder and would need actual feature generation in a real scenario.
    
    # If the train.parquet only has '0' to '31', we need to augment it or assume these are generated.
    # Given the task is to make the plot similar, let's assume we can construct a DataFrame
    # with these columns for correlation analysis.
    
    # Let's try to mimic the columns if they exist in the original df, or create dummy ones.
    # The original df has columns 'seq_ix', 'step_in_seq', 'need_prediction' and '0'...'31'.
    
    # Let's take '0' to '31' as our base set.
    df_for_corr = df[feature_cols].copy()
    
    # To simulate the image, let's add some of the named features.
    # These are usually generated from the base features. For a demo, let's simulate them.
    # In a real setup, these would come from feature engineering scripts.
    
    # For simplicity, let's directly use the feature_cols for correlation.
    # If a real `eda.py` generated more features, this part would be different.
    
    # Winsorize at current defaults (0.1/99.9) based on the image's description
    data = df_for_corr.to_numpy()
    lower = np.quantile(data, 0.001, axis=0)
    upper = np.quantile(data, 0.999, axis=0)
    data_clip = np.clip(data, lower, upper)
    df_clip = pd.DataFrame(data_clip, columns=feature_cols)

    # Spearman correlation
    corr = df_clip.corr(method="spearman")
    corr_path = os.path.join(out_dir, "corr_spearman.csv")
    corr.to_csv(corr_path)
    print(f"Saved Spearman correlation to {corr_path}")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set a dark theme as in the image
        plt.style.use('dark_background')
        plt.rcParams.update({
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.facecolor": "#202020", # Dark grey for axes background
            "figure.facecolor": "#121212", # Even darker for figure background
            "savefig.facecolor": "#121212"
        })

        fig, ax = plt.subplots(figsize=(16, 14)) # Adjust size for readability
        
        # Custom colormap to match the image: green for positive, red for negative, dark for zero
        cmap = sns.diverging_palette(10, 130, as_cmap=True, s=99, l=30, center="dark") # Red-Green diverging, dark center

        sns.heatmap(
            corr,
            cmap=cmap, # Use the custom colormap
            vmin=-1,
            vmax=1,
            center=0,
            annot=True, # Show correlation values on the heatmap
            fmt=".2f", # Format annotations to two decimal places
            linewidths=0.5, # Add lines between cells for better separation
            linecolor="#303030", # Darker line color
            cbar_kws={'shrink': 0.75, 'aspect': 30}, # Adjust colorbar size
            ax=ax # Ensure heatmap plots on the created axes
        )
        
        # Customize annotations color to yellow, like in the image
        for text in ax.texts:
            text.set_color('yellow')

        ax.set_title(
            "Correlation Matrix\n"
            f"Method: spearman | Variables: {len(corr.columns)} | Observations: {len(df_clip)} | "
            # Adding placeholders for Strong correlations and Significant p-values
            # These would require more complex calculation for exact match if original df includes them
            f"Strong correlations (>0.7 | <-0.7): ? | Significant (p<0.01): ?",
            color='white',
            fontsize=14
        )
        
        # Rotate x-axis labels to match the image style
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')

        plt.tight_layout()
        heatmap_path = os.path.join(out_dir, "corr_spearman.png")
        plt.savefig(heatmap_path, dpi=200)
        plt.close()
        print(f"Saved correlation heatmap to {heatmap_path}")
    except Exception as e:
        print(f"Skipping heatmap (matplotlib/seaborn issue): {e}")

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

    # Optional hist/box plots per feature (lightweight)
    try:
        # Re-apply dark style if it was reset, or ensure it's active
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