import pandas as pd
import numpy as np
import os

def main():
    # Load the raw correlation matrix (ordered by index 0-31)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))
    corr_path = os.path.join(repo_root, "outputs", "eda", "corr_spearman.csv")
    
    if not os.path.exists(corr_path):
        print("Correlation CSV not found. Please run scripts/eda_returns.py first.")
        return

    df_corr = pd.read_csv(corr_path, index_col=0)
    
    # Use absolute correlation
    df_abs = df_corr.abs()
    
    # Mask diagonal
    np.fill_diagonal(df_abs.values, 0)
    
    pairs = []
    dropped = set()
    
    print("--- High Correlation Pairs (> 0.90) ---")
    # Greedy approach: Find max corr, report, mark one for potential dropping
    # We iterate through the matrix upper triangle
    columns = df_corr.columns.tolist()
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_i = columns[i]
            col_j = columns[j]
            val = df_corr.iloc[i, j]
            
            if abs(val) >= 0.90:
                pairs.append((col_i, col_j, val))
                print(f"Pair ({col_i}, {col_j}): {val:.4f}")

    print("\n--- Proposed Feature Engineering Strategy (v21) ---")
    
    # 1. Spread Candidates (Very high correlation but not 1.0)
    print("1. Add 'Spread' features for these pairs (Likely stationary relationships):")
    spread_candidates = [p for p in pairs if abs(p[2]) > 0.95]
    for p in spread_candidates:
        print(f"   - spread_{p[0]}_{p[1]} = {p[0]} - ({np.sign(p[2])} * {p[1]})")

    # 2. Drop Candidates (Redundancy)
    # A simple heuristic: if A and B are >0.98 correlated, drop B.
    to_drop = []
    print("\n2. Drop Redundant features (Corr > 0.98):")
    for i in range(len(columns)):
        col_i = columns[i]
        if col_i in to_drop: 
            continue
        for j in range(i + 1, len(columns)):
            col_j = columns[j]
            if col_j in to_drop:
                continue
            if df_abs.iloc[i, j] > 0.98:
                print(f"   - Drop {col_j} (matches {col_i} with {df_corr.iloc[i, j]:.4f})")
                to_drop.append(col_j)
    
    if not to_drop:
        print("   (None found at > 0.98 threshold)")

if __name__ == "__main__":
    main()
