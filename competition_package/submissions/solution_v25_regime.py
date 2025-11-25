import os
import sys
import json
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

# --- Architectures ---

class LagMLP_v19(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x): return self.net(x)

class LagMLP_v21(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x): return self.net(x)

class RegimeClassifier(nn.Module):
    def __init__(self, input_dim, n_classes=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x): return self.net(x)

class PredictionModel:
    def __init__(self):
        self.device = torch.device("cpu")
        models_dir = os.path.join(BASE_DIR, "models")
        
        # --- 1. Load v19 (Level) ---
        norm_v19 = np.load(os.path.join(models_dir, "lag_mlp_normalization.npz"))
        self.x_mean_v19 = norm_v19["x_mean"].astype(np.float32)
        self.x_std_v19 = norm_v19["x_std"].astype(np.float32)
        self.clip_min_v19 = norm_v19["clip_min"].astype(np.float32)
        self.clip_max_v19 = norm_v19["clip_max"].astype(np.float32)
        
        self.extractor_v19 = FeatureExtractor(n_lags=10, clip_min=self.clip_min_v19, clip_max=self.clip_max_v19, use_spreads=False)
        self.models_v19 = self._load_ensemble(models_dir, "lag_mlp_fold", LagMLP_v19)

        # --- 2. Load v21 (Residual) ---
        norm_v21 = np.load(os.path.join(models_dir, "lag_mlp_v21_normalization.npz"))
        self.x_mean_v21 = norm_v21["x_mean"].astype(np.float32)
        self.x_std_v21 = norm_v21["x_std"].astype(np.float32)
        self.clip_min_v21 = norm_v21["clip_min"].astype(np.float32)
        self.clip_max_v21 = norm_v21["clip_max"].astype(np.float32)
        
        self.extractor_v21 = FeatureExtractor(n_lags=10, clip_min=self.clip_min_v21, clip_max=self.clip_max_v21, use_spreads=True)
        self.models_v21 = self._load_ensemble(models_dir, "lag_mlp_v21_fold", LagMLP_v21)
        
        # --- 3. Load Regime Classifier ---
        # Classifier uses v19 normalization
        self.classifier = RegimeClassifier(input_dim=1185, n_classes=5) # v19 dim
        clf_path = os.path.join(models_dir, "regime_classifier.pth")
        self.classifier.load_state_dict(torch.load(clf_path, map_location=self.device))
        self.classifier.eval()
        
        torch.set_num_threads(1)
        self.current_seq = None

    def _load_ensemble(self, models_dir, prefix, model_cls):
        models = []
        files = sorted([f for f in os.listdir(models_dir) if f.startswith(prefix) and f.endswith(".pth")])
        if not files: raise FileNotFoundError(f"No models for {prefix}")
        
        ckpt0 = torch.load(os.path.join(models_dir, files[0]), map_location=self.device)
        input_dim = ckpt0["input_dim"]
        hidden_dim = ckpt0["hidden_dim"]
        output_dim = ckpt0["output_dim"]
        
        for f in files:
            path = os.path.join(models_dir, f)
            ckpt = torch.load(path, map_location=self.device)
            m = model_cls(input_dim, hidden_dim, output_dim)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            models.append(m)
        return models

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if data_point.seq_ix != self.current_seq:
            self.current_seq = data_point.seq_ix
            self.extractor_v19.reset()
            self.extractor_v21.reset()

        # Stream features
        feats_v19 = self.extractor_v19.stream(data_point.state, data_point.step_in_seq, data_point.seq_ix)
        feats_v21 = self.extractor_v21.stream(data_point.state, data_point.step_in_seq, data_point.seq_ix)
        
        if not data_point.need_prediction: return None
        if feats_v19 is None: return data_point.state.astype(np.float32) # Fallback
        
        # Normalize
        x_v19_norm = (feats_v19 - self.x_mean_v19) / self.x_std_v19
        x_v21_norm = (feats_v21 - self.x_mean_v21) / self.x_std_v21
        
        t_v19 = torch.from_numpy(x_v19_norm).unsqueeze(0)
        t_v21 = torch.from_numpy(x_v21_norm).unsqueeze(0)
        
        # 1. Detect Regime (using v19 normalized features)
        with torch.no_grad():
            logits = self.classifier(t_v19)
            regime = torch.argmax(logits, dim=1).item()
            
        # 2. Select Alpha Strategy
        if regime == 3:
            # Cluster 3 (High Volatility): Residual model fails hard. Use Level.
            alpha = 1.0
        elif regime == 2:
            # Cluster 2 (Trending): Residual model wins slightly.
            alpha = 0.3 # Bias towards residual (v21)
        else:
            # Default (Clusters 0, 1, 4): Robust blend
            alpha = 0.55
            
        # 3. Inference
        pred_v19 = np.zeros(32, dtype=np.float32)
        pred_v21_resid = np.zeros(32, dtype=np.float32)
        
        with torch.no_grad():
            if alpha > 0.0:
                for m in self.models_v19: pred_v19 += m(t_v19).squeeze(0).numpy()
                pred_v19 /= len(self.models_v19)
            
            if alpha < 1.0:
                for m in self.models_v21: pred_v21_resid += m(t_v21).squeeze(0).numpy()
                pred_v21_resid /= len(self.models_v21)
        
        pred_v21 = data_point.state.astype(np.float32) + pred_v21_resid
        
        # 4. Blend
        final_pred = alpha * pred_v19 + (1.0 - alpha) * pred_v21
        return final_pred
