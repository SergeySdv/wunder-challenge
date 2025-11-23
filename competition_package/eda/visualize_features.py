"""
Visualize all 32 features from the dataset.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_all_features(df, seq_ix=0, max_steps=1000):
    """Plot all 32 features for a given sequence."""

    # Filter to single sequence
    seq_data = df[df['seq_ix'] == seq_ix].head(max_steps).copy()
    feature_cols = [str(i) for i in range(32)]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(8, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Plot each feature in a subplot
    for i, col in enumerate(feature_cols):
        row = i // 4
        col_idx = i % 4
        ax = fig.add_subplot(gs[row, col_idx])

        ax.plot(seq_data['step_in_seq'], seq_data[col], linewidth=0.8, alpha=0.8)
        ax.set_title(f'Feature {col}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        # Add mean line
        mean_val = seq_data[col].mean()
        ax.axhline(mean_val, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    fig.suptitle(f'All 32 Features - Sequence {seq_ix}', fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), f'plots_features_seq{seq_ix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_features_overlay(df, seq_ix=0, max_steps=1000):
    """Plot all features overlaid to see relative patterns."""

    seq_data = df[df['seq_ix'] == seq_ix].head(max_steps).copy()
    feature_cols = [str(i) for i in range(32)]

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot all features with different colors
    for i, col in enumerate(feature_cols):
        ax.plot(seq_data['step_in_seq'], seq_data[col],
                linewidth=0.6, alpha=0.5, label=f'F{col}')

    ax.set_title(f'All 32 Features Overlaid - Sequence {seq_ix}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Step in Sequence', fontsize=12)
    ax.set_ylabel('Standardized Value', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add legend outside plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7, ncol=2)

    output_path = os.path.join(os.path.dirname(__file__), f'plots_features_overlay_seq{seq_ix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_feature_statistics(df):
    """Plot statistical comparisons across all 32 features."""

    feature_cols = [str(i) for i in range(32)]

    # Compute statistics
    stats = {
        'mean': [],
        'std': [],
        'autocorr_lag1': [],
        'var_ratio': []  # var(diff) / var(level)
    }

    for col in feature_cols:
        stats['mean'].append(df[col].mean())
        stats['std'].append(df[col].std())

        # Autocorrelation lag-1 (averaged across all sequences)
        autocorrs = []
        for seq_ix in df['seq_ix'].unique():
            seq_vals = df[df['seq_ix'] == seq_ix][col].values
            if len(seq_vals) > 1:
                autocorrs.append(np.corrcoef(seq_vals[:-1], seq_vals[1:])[0, 1])
        stats['autocorr_lag1'].append(np.mean(autocorrs))

        # Variance ratio
        var_level = df[col].var()
        var_diff = df[col].diff().var()
        stats['var_ratio'].append(var_diff / var_level if var_level > 0 else np.nan)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Feature Statistics Comparison (All 32 Features)', fontsize=14, fontweight='bold')

    # Plot 1: Mean values
    axes[0, 0].bar(range(32), stats['mean'], color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Mean Value per Feature')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Mean')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)

    # Plot 2: Standard deviation
    axes[0, 1].bar(range(32), stats['std'], color='green', alpha=0.7)
    axes[0, 1].set_title('Standard Deviation per Feature')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(1.0, color='red', linestyle='--', linewidth=1, label='Expected (standardized)')
    axes[0, 1].legend()

    # Plot 3: Lag-1 autocorrelation (KEY FOR PRICE DETECTION!)
    axes[1, 0].bar(range(32), stats['autocorr_lag1'], color='orange', alpha=0.7)
    axes[1, 0].set_title('Lag-1 Autocorrelation (Higher = More Price-Like)')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0.5, color='red', linestyle='--', linewidth=1, label='Threshold (0.5)')
    axes[1, 0].legend()

    # Highlight highest autocorr
    max_autocorr_idx = np.argmax(stats['autocorr_lag1'])
    axes[1, 0].bar(max_autocorr_idx, stats['autocorr_lag1'][max_autocorr_idx],
                   color='red', alpha=0.9, label=f'Max: F{max_autocorr_idx}')

    # Plot 4: Variance ratio (lower = more price-like)
    axes[1, 1].bar(range(32), stats['var_ratio'], color='purple', alpha=0.7)
    axes[1, 1].set_title('Variance Ratio: Var(diff)/Var(level) (Lower = More Price-Like)')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Variance Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0.1, color='red', linestyle='--', linewidth=1, label='Typical price (<0.1)')
    axes[1, 1].legend()

    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'plots_feature_statistics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

    # Print summary
    print("\n" + "="*80)
    print("FEATURE STATISTICS SUMMARY")
    print("="*80)
    print(f"\nHighest Lag-1 Autocorrelation (most price-like):")
    autocorr_sorted = sorted(enumerate(stats['autocorr_lag1']), key=lambda x: x[1], reverse=True)
    for idx, (feat_idx, autocorr) in enumerate(autocorr_sorted[:5]):
        print(f"  {idx+1}. Feature {feat_idx:2d}: {autocorr:.4f}")

    print(f"\nLowest Variance Ratio (most price-like):")
    var_ratio_sorted = sorted(enumerate(stats['var_ratio']), key=lambda x: x[1])
    for idx, (feat_idx, var_ratio) in enumerate(var_ratio_sorted[:5]):
        print(f"  {idx+1}. Feature {feat_idx:2d}: {var_ratio:.4f}")
    print("="*80)

    return stats


def plot_multiple_sequences_comparison(df, feature_idx=3, n_sequences=5):
    """Plot the same feature across multiple sequences to see variation."""

    fig, axes = plt.subplots(n_sequences, 1, figsize=(16, 3*n_sequences), sharex=True)

    for i, seq_ix in enumerate(df['seq_ix'].unique()[:n_sequences]):
        seq_data = df[df['seq_ix'] == seq_ix].copy()

        axes[i].plot(seq_data['step_in_seq'], seq_data[str(feature_idx)],
                    linewidth=0.8, color='steelblue')
        axes[i].set_title(f'Feature {feature_idx} - Sequence {seq_ix}', fontsize=10)
        axes[i].set_ylabel('Value', fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    axes[-1].set_xlabel('Step in Sequence', fontsize=10)
    fig.suptitle(f'Feature {feature_idx} Across Multiple Sequences', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), f'plots_feature{feature_idx}_multi_seq.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_all_features_multi_sequence(df, n_sequences=100, max_steps=1000):
    """Plot all 32 features with multiple sequences overlaid to show variation."""

    feature_cols = [str(i) for i in range(32)]
    sequences = df['seq_ix'].unique()[:n_sequences]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(8, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Plot each feature in a subplot
    for i, col in enumerate(feature_cols):
        row = i // 4
        col_idx = i % 4
        ax = fig.add_subplot(gs[row, col_idx])

        # Plot each sequence with transparency
        for seq_ix in sequences:
            seq_data = df[df['seq_ix'] == seq_ix].head(max_steps)
            ax.plot(seq_data['step_in_seq'], seq_data[col],
                   linewidth=0.3, alpha=0.15, color='steelblue')

        ax.set_title(f'Feature {col} ({n_sequences} sequences)',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        # Add mean line at 0
        ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    fig.suptitle(f'All 32 Features - {n_sequences} Sequences Overlaid',
                fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__),
                              f'plots_features_multi_{n_sequences}seq.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Create plots directory if needed
    print("\n📊 Generating visualizations...")

    # 1. Grid of all features for sequence 0
    print("\n1. Plotting all 32 features (grid view) for seq_ix=0...")
    plot_all_features(df, seq_ix=0)

    # 2. Overlay of all features
    print("2. Plotting all features overlaid for seq_ix=0...")
    plot_features_overlay(df, seq_ix=0)

    # 3. Statistical comparison
    print("3. Computing and plotting feature statistics...")
    stats = plot_feature_statistics(df)

    # 4. Multi-sequence comparison for feature 3 (suspected price)
    print("4. Plotting Feature 3 across multiple sequences...")
    plot_multiple_sequences_comparison(df, feature_idx=3, n_sequences=5)

    # 5. Also plot feature 0 (used in initial experiments)
    print("5. Plotting Feature 0 across multiple sequences...")
    plot_multiple_sequences_comparison(df, feature_idx=0, n_sequences=5)

    # 6. NEW: All features with 100 sequences overlaid
    print("6. Plotting all 32 features with 100 sequences overlaid...")
    plot_all_features_multi_sequence(df, n_sequences=100)

    print("\n✅ All visualizations complete!")
    print(f"\nPlots saved in: {base_dir}")
