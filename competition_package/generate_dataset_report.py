"""
Generate a markdown report of the dataset for sharing with remote agents.
"""
import os
import pandas as pd
import numpy as np


def generate_markdown_report(dataset_path: str, output_path: str, max_rows_sample: int = 20) -> None:
    """Generate a comprehensive markdown report of the dataset."""

    df = pd.read_parquet(dataset_path)

    with open(output_path, 'w') as f:
        f.write("# Wunder Challenge Dataset Report\n\n")
        f.write(f"**Generated from:** `{dataset_path}`\n\n")
        f.write("---\n\n")

        # 1. Basic Info
        f.write("## 1. Dataset Overview\n\n")
        f.write(f"- **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns\n")
        f.write(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        f.write(f"- **Number of Sequences:** {df['seq_ix'].nunique()}\n")
        f.write(f"- **Steps per Sequence:** {df.groupby('seq_ix').size().iloc[0]} (uniform)\n\n")

        # 2. Column Info
        f.write("## 2. Columns\n\n")
        f.write("| Column | Type | Non-Null | Unique Values | Description |\n")
        f.write("|--------|------|----------|---------------|-------------|\n")

        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            nunique = df[col].nunique()

            if col == 'seq_ix':
                desc = "Sequence identifier (0-516)"
            elif col == 'step_in_seq':
                desc = "Step within sequence (0-999)"
            elif col == 'need_prediction':
                desc = "1 if prediction required, 0 otherwise"
            else:
                desc = f"Feature {col}"

            f.write(f"| `{col}` | {dtype} | {non_null:,} | {nunique:,} | {desc} |\n")

        f.write("\n")

        # 3. Sample Data
        f.write("## 3. Sample Data (First 20 Rows)\n\n")
        sample_df = df.head(max_rows_sample)
        f.write(sample_df.to_markdown(index=False))
        f.write("\n\n")

        # 4. Statistics for Meta Columns
        f.write("## 4. Meta Column Statistics\n\n")
        f.write("### seq_ix\n\n")
        f.write(f"- **Range:** {df['seq_ix'].min()} to {df['seq_ix'].max()}\n")
        f.write(f"- **Unique sequences:** {df['seq_ix'].nunique()}\n\n")

        f.write("### step_in_seq\n\n")
        f.write(f"- **Range:** {df['step_in_seq'].min()} to {df['step_in_seq'].max()}\n")
        f.write(f"- **Unique steps:** {df['step_in_seq'].nunique()}\n\n")

        f.write("### need_prediction\n\n")
        need_pred_counts = df['need_prediction'].value_counts().sort_index()
        f.write("| Value | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        for val, count in need_pred_counts.items():
            pct = count / len(df) * 100
            f.write(f"| {val} | {count:,} | {pct:.2f}% |\n")
        f.write("\n")

        # 5. Feature Statistics
        f.write("## 5. Feature Column Statistics (0-31)\n\n")
        feature_cols = [str(i) for i in range(32)]
        stats_df = df[feature_cols].describe()

        f.write("### Summary Statistics\n\n")
        f.write(stats_df.to_markdown())
        f.write("\n\n")

        # 6. Feature Correlations (sample)
        f.write("## 6. Feature Correlation Matrix (First 10 Features)\n\n")
        corr_matrix = df[feature_cols[:10]].corr()
        f.write(corr_matrix.to_markdown())
        f.write("\n\n")

        # 7. Sample Sequence Analysis
        f.write("## 7. Sample Sequence (seq_ix=0)\n\n")
        sample_seq = df[df['seq_ix'] == 0].head(20)
        f.write(sample_seq.to_markdown(index=False))
        f.write("\n\n")

        # 8. Missing Values
        f.write("## 8. Missing Values\n\n")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            f.write("✅ **No missing values found in the dataset.**\n\n")
        else:
            f.write("| Column | Missing Count | Percentage |\n")
            f.write("|--------|---------------|------------|\n")
            for col in missing[missing > 0].index:
                count = missing[col]
                pct = count / len(df) * 100
                f.write(f"| `{col}` | {count:,} | {pct:.2f}% |\n")
            f.write("\n")

        # 9. Data Distribution Info
        f.write("## 9. Value Distribution (First 5 Features)\n\n")
        for i in range(min(5, len(feature_cols))):
            col = feature_cols[i]
            f.write(f"### Feature {col}\n\n")
            f.write(f"- **Min:** {df[col].min():.4f}\n")
            f.write(f"- **Max:** {df[col].max():.4f}\n")
            f.write(f"- **Mean:** {df[col].mean():.4f}\n")
            f.write(f"- **Median:** {df[col].median():.4f}\n")
            f.write(f"- **Std Dev:** {df[col].std():.4f}\n")

            # Percentiles
            percentiles = df[col].quantile([0.25, 0.5, 0.75, 0.95, 0.99])
            f.write(f"- **25th percentile:** {percentiles[0.25]:.4f}\n")
            f.write(f"- **75th percentile:** {percentiles[0.75]:.4f}\n")
            f.write(f"- **95th percentile:** {percentiles[0.95]:.4f}\n")
            f.write(f"- **99th percentile:** {percentiles[0.99]:.4f}\n\n")

        # 10. Notes
        f.write("## 10. Important Notes\n\n")
        f.write("- **Prediction Task:** At each step where `need_prediction == 1`, predict the next 32-dimensional state vector.\n")
        f.write("- **Evaluation Metric:** R² (coefficient of determination) averaged across all 32 features.\n")
        f.write("- **Warm-up Period:** Steps 0-99 in each sequence are for context building (no scoring).\n")
        f.write("- **Scored Steps:** Steps 100-998 (step 999 is the last state, no next state to predict).\n")
        f.write("- **Sequence Independence:** Each `seq_ix` is completely independent; reset model state on sequence change.\n\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")
    output_path = os.path.join(base_dir, "datasets", "DATASET_REPORT.md")

    print(f"Loading dataset from: {dataset_path}")
    generate_markdown_report(dataset_path, output_path)
    print(f"\n✅ Dataset report generated: {output_path}")
    print(f"📄 You can now share this markdown file with remote agents!")
