"""
Utility to dump CatBoost feature importances to CSV for a saved v19-model.

Usage:
  python dump_catboost_importance.py --model_path path/to/cat_model.cbm --out_path cat_importance.csv

Notes:
- This does NOT train; it only loads an existing CatBoost model.
- Feature ordering matches v19 build_dataset in train_catboost_experiment.py:
  lags (10x32), deltas (10x32), mean, std, ac1/ac2/ac3, acf_sum, frac_above_mean,
  q25/median/q75, iqr, skew, kurt, cv, slope, r2, curvature, step_val.
"""

import argparse
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def build_feature_names() -> list[str]:
    names = []
    # lags
    for k in range(10):
        for f in range(32):
            names.append(f"lag[{k}]/feat{f}")
    # deltas
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
        ("curve", 32),
        ("step_val", 1),
    ]
    for name, dim in blocks:
        if dim == 1:
            names.append(name)
        else:
            for f in range(dim):
                names.append(f"{name}/feat{f}")
    return names


def main():
    parser = argparse.ArgumentParser(description="Dump CatBoost feature importances to CSV.")
    parser.add_argument("--model_path", required=True, help="Path to saved CatBoost .cbm model.")
    parser.add_argument("--out_path", default="cat_importance.csv", help="Output CSV path.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    print(f"Loading model from {args.model_path} ...")
    model = CatBoostRegressor()
    model.load_model(args.model_path)

    print("Computing feature importances...")
    importances = model.get_feature_importance()
    names = build_feature_names()
    if len(importances) != len(names):
        raise ValueError(f"Mismatch: importances {len(importances)} vs names {len(names)}")

    df = pd.DataFrame({"feature": names, "importance": importances})
    df_sorted = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(args.out_path, index=False)
    print(f"Saved importances to {args.out_path} (top 10):")
    print(df_sorted.head(10))


if __name__ == "__main__":
    main()
