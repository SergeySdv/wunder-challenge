import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from train_model import load_dataset


def _import_catch22():
    """Import catch22_all from the catch22 package, with a clear error if missing."""
    try:
        from catch22 import catch22_all
    except ImportError as exc:  # pragma: no cover - simple import guard
        raise ImportError(
            "The 'catch22' package is required to run this script.\n"
            "Install it inside your virtualenv, e.g.:\n"
            "  source .venv/bin/activate\n"
            "  pip install catch22"
        ) from exc
    return catch22_all


def compute_catch22_for_sequence(
    df_seq: pd.DataFrame,
    feature_cols: List[str],
    catch22_all,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute catch22 features for one sequence and all feature columns.

    Returns
    -------
    values : np.ndarray
        Array of shape (n_dims, 22) with catch22 feature values.
    names : list of str
        Names of the 22 catch22 features (same for all dimensions).
    """
    n_dims = len(feature_cols)
    values = np.zeros((n_dims, 22), dtype=np.float32)
    feature_names: List[str] = []

    for j, col in enumerate(feature_cols):
        series = df_seq[col].values.astype(np.float64)
        res = catch22_all(series)
        # res is expected to have keys 'names' and 'values'
        if not feature_names:
            feature_names = list(res.get("names", []))
        vals = np.array(res.get("values", []), dtype=np.float32)
        if vals.shape[0] != 22:
            raise ValueError(
                f"Expected 22 catch22 values for column {col}, got shape {vals.shape}"
            )
        values[j] = vals

    return values, feature_names


def main() -> None:
    """
    Compute per-sequence, per-dimension catch22 features and save to an .npz file.

    Output file: datasets/catch22_per_seq.npz
    Contains:
      - seq_ids: np.ndarray of shape (n_seqs,)
      - feature_cols: np.ndarray of shape (n_dims,)
      - catch22_names: np.ndarray of shape (22,)
      - catch22_values: np.ndarray of shape (n_seqs, n_dims, 22)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "datasets", "train.parquet")
    output_path = os.path.join(base_dir, "datasets", "catch22_per_seq.npz")

    print(f"Loading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)
    print(f"Full dataset shape: {df.shape}")

    # Identify state feature columns (same convention as train_model.py)
    feature_cols = [
        c for c in df.columns if c not in ("seq_ix", "step_in_seq", "need_prediction")
    ]
    print(f"Using {len(feature_cols)} feature columns: {feature_cols}")

    catch22_all = _import_catch22()

    seq_ids = sorted(df["seq_ix"].unique())
    n_seqs = len(seq_ids)
    n_dims = len(feature_cols)
    print(f"Found {n_seqs} sequences.")

    # Will fill catch22_values[seq_idx, dim_idx, feature_idx]
    catch22_values = np.zeros((n_seqs, n_dims, 22), dtype=np.float32)
    catch22_names: List[str] = []

    for i, seq_ix in enumerate(seq_ids):
        df_seq = (
            df[df["seq_ix"] == seq_ix]
            .sort_values("step_in_seq")
            .reset_index(drop=True)
        )
        vals, names = compute_catch22_for_sequence(df_seq, feature_cols, catch22_all)
        catch22_values[i] = vals
        if not catch22_names:
            catch22_names = names

        if (i + 1) % 10 == 0 or i == n_seqs - 1:
            print(f"Processed {i + 1}/{n_seqs} sequences...", flush=True)

    np.savez(
        output_path,
        seq_ids=np.array(seq_ids, dtype=np.int64),
        feature_cols=np.array(feature_cols, dtype=object),
        catch22_names=np.array(catch22_names, dtype=object),
        catch22_values=catch22_values,
    )

    print(f"Saved catch22 features to: {output_path}")


if __name__ == "__main__":
    main()

