import os

import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "datasets", "train.parquet")

    print(f"Loading dataset from: {data_path}")
    df = pd.read_parquet(data_path)

    print("\n=== Basic info ===")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"  [{i:03}] {col}")

    print("\nDtypes:")
    print(df.dtypes)

    print("\n=== Head (first 10 rows) ===")
    print(df.head(10))

    print("\n=== Key columns summary ===")
    if {"seq_ix", "step_in_seq", "need_prediction"}.issubset(df.columns):
        print("\nValue counts for need_prediction:")
        print(df["need_prediction"].value_counts(dropna=False))

        print("\nstep_in_seq: min, max, unique count")
        print(
            df["step_in_seq"].min(),
            df["step_in_seq"].max(),
            df["step_in_seq"].nunique(),
        )

        print("\nNumber of unique sequences (seq_ix):")
        print(df["seq_ix"].nunique())

        print("\nExample seq_ix sample (first 5):")
        print(df["seq_ix"].drop_duplicates().head())
    else:
        print("Expected columns seq_ix, step_in_seq, need_prediction not all present.")

    feature_cols = df.columns[3:]
    if len(feature_cols) > 0:
        sample_features = feature_cols[: min(10, len(feature_cols))]
        print("\n=== Feature columns (first 10) ===")
        for col in sample_features:
            print(f"- {col}")

        print("\n=== Descriptive stats for sample features ===")
        print(df[sample_features].describe().T)
    else:
        print("\nNo feature columns detected beyond metadata columns.")


if __name__ == "__main__":
    main()

