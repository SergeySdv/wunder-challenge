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

# --- Model Architecture (Must match train_feature_gru.py) ---
class FeatureGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=32, layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        
        self.head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, h=None):
        # x: (Batch, Seq, Input_Dim)
        encoded = self.encoder(x) # (Batch, Seq, Hidden)
        out, h_new = self.gru(encoded, h)
        pred = self.head(out)
        return pred, h_new

class PredictionModel:
    def __init__(self):
        self.device = torch.device("cpu")
        
        # Path setup
        models_dir = os.path.join(BASE_DIR, "models")
        
        # Load Normalization
        norm_path = os.path.join(models_dir, "feature_gru_normalization.npz")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Normalization file not found: {norm_path}")
            
        norm = np.load(norm_path)
        self.x_mean = norm["x_mean"].astype(np.float32)
        self.x_std = norm["x_std"].astype(np.float32)
        self.clip_min = norm["clip_min"].astype(np.float32)
        self.clip_max = norm["clip_max"].astype(np.float32)
        
        # Config (Fixed from training script)
        self.n_lags = 10
        self.input_dim = 1187
        
        # Initialize Feature Extractor
        self.extractor = FeatureExtractor(
            n_lags=self.n_lags, 
            clip_min=self.clip_min, 
            clip_max=self.clip_max,
            use_spreads=True
        )
        
        # Load Model
        model_path = os.path.join(models_dir, "feature_gru_best.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = FeatureGRU(input_dim=self.input_dim, hidden_dim=256, output_dim=32, layers=2, dropout=0.0)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
            
        torch.set_num_threads(1)
        self.current_seq = None
        self.hidden_state = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Reset if new sequence
        if data_point.seq_ix != self.current_seq:
            self.current_seq = data_point.seq_ix
            self.extractor.reset()
            self.hidden_state = None

        # Update extractor state
        # extractor.stream returns features if ready, else None
        # For GRU, we need to feed SOMETHING every step to keep state updated.
        features = self.extractor.stream(
            data_point.state, 
            data_point.step_in_seq, 
            data_point.seq_ix
        )
        
        if features is None:
            # Warmup phase (steps 0-9):
            # Feed zeros to initialize RNN flow
            # Note: This effectively tells the RNN "no features yet", 
            # allowing it to spin up its internal state (e.g. biases).
            features_norm = np.zeros(self.input_dim, dtype=np.float32)
        else:
            # Normalize valid features
            features_norm = (features - self.x_mean) / self.x_std
            
        # Inference Step (Stateful)
        # Input: (Batch=1, Seq=1, Dim)
        x_tensor = torch.from_numpy(features_norm).float().unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            # Pass hidden state from previous step, get new one
            pred_residual_tensor, self.hidden_state = self.model(x_tensor, self.hidden_state)
            pred_residual = pred_residual_tensor.squeeze().numpy()
        
        if not data_point.need_prediction:
            return None
            
        # If features were None (steps 0-9), we technically shouldn't be predicting.
        # But need_prediction is False for those steps anyway.
        # If need_prediction is True (step 100+), features should be valid.
        
        # Reconstruct Level (Target was y_{t+1} - y_t)
        # pred_{t+1} = y_t + residual
        pred_level = data_point.state.astype(np.float32) + pred_residual
        
        return pred_level
