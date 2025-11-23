"""
Diagnostic: evaluate current solution model R² on early vs late steps within each sequence.
Splits by step_in_seq: early <= 699, late >= 700 (inclusive).

Usage (from repo root):
  cd competition_package
  ../.venv/bin/python scripts/diagnostic_step_splits.py

This does not retrain; it scores the existing solution.PredictionModel over train.parquet and
reports overall, early, and late R² to check for drift.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ensure local utils is used
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils import DataPoint  # noqa: E402


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    dataset_path = os.path.join(root_dir, "datasets", "train.parquet")

    # Import the current solution model
    from solution import PredictionModel  # noqa: E402

    df = pd.read_parquet(dataset_path)
    model = PredictionModel()

    y_true = []
    y_pred = []
    steps = []

    for _, row in df.iterrows():
        dp = DataPoint(
            seq_ix=int(row["seq_ix"]),
            step_in_seq=int(row["step_in_seq"]),
            need_prediction=bool(row["need_prediction"]),
            state=row[[str(i) for i in range(32)]].to_numpy(dtype=np.float32),
        )
        pred = model.predict(dp)
        if pred is None:
            continue
        y_true.append(row[[str(i) for i in range(32)]].to_numpy(dtype=np.float32))
        y_pred.append(pred.astype(np.float32))
        steps.append(dp.step_in_seq)

    y_true = np.stack(y_true)
    y_pred = np.stack(y_pred)
    steps = np.array(steps)

    def safe_r2(y_t, y_p):
        r2s = [r2_score(y_t[:, i], y_p[:, i]) for i in range(y_t.shape[1])]
        return float(np.mean(r2s))

    overall_r2 = safe_r2(y_true, y_pred)
    mask_early = steps <= 699
    mask_late = steps >= 700

    early_r2 = safe_r2(y_true[mask_early], y_pred[mask_early]) if mask_early.any() else float("nan")
    late_r2 = safe_r2(y_true[mask_late], y_pred[mask_late]) if mask_late.any() else float("nan")

    print(f"Overall R²: {overall_r2:.5f}")
    print(f"Early (<=699) R²: {early_r2:.5f}")
    print(f"Late (>=700) R²: {late_r2:.5f}")


if __name__ == "__main__":
    main()
