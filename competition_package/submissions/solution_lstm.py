import os
import json
import numpy as np
import torch
import torch.nn as nn
from utils import DataPoint


# Paths (relative to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_submission.pth")
NORM_PATH = os.path.join(BASE_DIR, "models", "lstm_submission_norm.npz")
META_PATH = os.path.join(BASE_DIR, "models", "lstm_submission_meta.json")


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        return self.head(h_last)


class PredictionModel:
    def __init__(self):
        # Load normalization / config
        norm = np.load(NORM_PATH)
        self.clip_min = norm["clip_min"].astype(np.float32)
        self.clip_max = norm["clip_max"].astype(np.float32)
        self.mean = norm["mean"].astype(np.float32)
        self.std = norm["std"].astype(np.float32)
        self.window = int(norm["window"])

        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                meta = json.load(f)
            hidden = int(meta.get("hidden", 128))
            layers = int(meta.get("layers", 2))
            dropout = float(meta.get("dropout", 0.1))
        else:
            hidden, layers, dropout = 128, 2, 0.1

        self.model = LSTMRegressor(input_dim=32, hidden_size=hidden, num_layers=layers, dropout=dropout)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        torch.set_num_threads(max(1, os.cpu_count() or 1))

        self.buffer = []  # list of np arrays (32,)
        self.current_seq = None

    def _reset_state(self):
        self.buffer = []

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if data_point.seq_ix != self.current_seq:
            self.current_seq = data_point.seq_ix
            self._reset_state()

        state = data_point.state.astype(np.float32)
        # update buffer
        self.buffer.append(state)
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window :]

        if not data_point.need_prediction:
            return None

        if len(self.buffer) < self.window:
            # fallback: persistence
            return state.copy()

        # build window tensor
        window_arr = np.stack(self.buffer, axis=0)
        window_arr = np.clip(window_arr, self.clip_min, self.clip_max)
        window_arr = (window_arr - self.mean) / self.std
        x = torch.from_numpy(window_arr).unsqueeze(0).to(self.device)  # (1, T, 32)

        with torch.no_grad():
            pred = self.model(x).cpu().numpy()[0]

        return pred.astype(np.float32)


if __name__ == "__main__":
    # Optional: quick streaming eval on train.parquet
    from utils import ScorerStepByStep
    import pandas as pd

    df = pd.read_parquet(os.path.join(BASE_DIR, "datasets", "train.parquet"))
    scorer = ScorerStepByStep(PredictionModel())
    r2 = scorer.score_dataframe(df)
    print(f"Mean R2: {r2:.5f}")
