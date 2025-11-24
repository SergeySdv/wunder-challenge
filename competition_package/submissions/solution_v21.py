import os
import sys
import numpy as np
import torch
from torch import nn
from typing import Optional

# Add current directory to sys.path to allow importing src if zipped at root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils import DataPoint

# Try importing FeatureExtractor from src
try:
    from src.features.extractor import FeatureExtractor
except ImportError:
    # If src is not in path, try appending it
    sys.path.append(os.path.join(BASE_DIR, "src"))
    from src.features.extractor import FeatureExtractor

# --- Model Architecture (Must match train_mlp_v21.py) ---
DROPOUT = 0.25

class LagMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class PredictionModel:
    def __init__(self):
        self.models = []
        self.device = torch.device("cpu")
        
        # Path setup
        models_dir = os.path.join(BASE_DIR, "models")
        
        # Load Normalization
        norm_path = os.path.join(models_dir, "lag_mlp_v21_normalization.npz")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Normalization file not found: {norm_path}")
            
        norm = np.load(norm_path)
        self.x_mean = norm["x_mean"].astype(np.float32)
        self.x_std = norm["x_std"].astype(np.float32)
        self.clip_min = norm["clip_min"].astype(np.float32)
        self.clip_max = norm["clip_max"].astype(np.float32)
        self.n_lags = int(norm["n_lags"])
        # Check if spread usage is recorded, otherwise default to True for v21
        self.use_spreads = bool(norm["use_spreads"]) if "use_spreads" in norm else True
        
        # Initialize Feature Extractor
        self.extractor = FeatureExtractor(
            n_lags=self.n_lags, 
            clip_min=self.clip_min, 
            clip_max=self.clip_max,
            use_spreads=self.use_spreads
        )
        
        # Load Ensemble Models
        fold_files = sorted([f for f in os.listdir(models_dir) if f.startswith("lag_mlp_v21_fold") and f.endswith(".pth")])
        if not fold_files:
            raise FileNotFoundError(f"No model weights found in {models_dir}")
            
        # Read architecture from first model
        first_ckpt = torch.load(os.path.join(models_dir, fold_files[0]), map_location=self.device)
        input_dim = first_ckpt["input_dim"]
        hidden_dim = first_ckpt["hidden_dim"]
        output_dim = first_ckpt["output_dim"]
        
        for fname in fold_files:
            path = os.path.join(models_dir, fname)
            ckpt = torch.load(path, map_location=self.device)
            model = LagMLP(input_dim, hidden_dim, output_dim)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.models.append(model)
            
        torch.set_num_threads(1)
        self.current_seq = None
        self.buffer = [] # Just to track length for safety, mostly handled by extractor

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Reset if new sequence
        if data_point.seq_ix != self.current_seq:
            self.current_seq = data_point.seq_ix
            self.extractor.reset()
            self.buffer = []

        # Update extractor state
        # extractor.stream returns features if ready, else None
        # It handles buffering internally.
        features = self.extractor.stream(
            data_point.state, 
            data_point.step_in_seq, 
            data_point.seq_ix
        )
        
        # Also keep a local buffer just to fallback if needed (persistence)
        # though extractor returns None if not ready.
        self.buffer.append(data_point.state)
        
        if not data_point.need_prediction:
            return None
            
        if features is None:
            # Not enough history yet (warmup)
            # Return current state as fallback (persistence)
            return data_point.state.astype(np.float32)
            
        # Normalize
        features_norm = (features - self.x_mean) / self.x_std
        
        # Inference
        x_tensor = torch.from_numpy(features_norm).unsqueeze(0) # (1, dim)
        
        preds_accum = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            for model in self.models:
                preds_accum += model(x_tensor).squeeze(0).numpy()
        
        pred_residual = preds_accum / len(self.models)
        
        # Reconstruct Level (Target was y_{t+1} - y_t)
        # pred_{t+1} = y_t + residual
        pred_level = data_point.state.astype(np.float32) + pred_residual
        
        return pred_level
