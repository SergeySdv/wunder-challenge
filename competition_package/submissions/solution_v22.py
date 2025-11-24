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

# --- Model Architectures ---

# LagMLP v19 (Dropout 0.2)
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

    def forward(self, x):
        return self.net(x)

# LagMLP v21 (Dropout 0.25)
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

    def forward(self, x):
        return self.net(x)

class PredictionModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.models_v19 = []
        self.models_v21 = []
        
        models_dir = os.path.join(BASE_DIR, "models")
        
        # --- Load Alphas ---
        alpha_path = os.path.join(models_dir, "alpha_blend_v22.npy")
        if not os.path.exists(alpha_path):
            raise FileNotFoundError(f"Alpha vector not found: {alpha_path}")
        self.alphas = np.load(alpha_path).astype(np.float32) # (32,)
        
        # --- Load v19 (Level) ---
        norm_v19_path = os.path.join(models_dir, "lag_mlp_normalization.npz")
        self.norm_v19 = np.load(norm_v19_path)
        self.x_mean_v19 = self.norm_v19["x_mean"].astype(np.float32)
        self.x_std_v19 = self.norm_v19["x_std"].astype(np.float32)
        self.clip_min_v19 = self.norm_v19["clip_min"].astype(np.float32)
        self.clip_max_v19 = self.norm_v19["clip_max"].astype(np.float32)
        self.n_lags_v19 = int(self.norm_v19["n_lags"])
        
        self.extractor_v19 = FeatureExtractor(
            n_lags=self.n_lags_v19, 
            clip_min=self.clip_min_v19, 
            clip_max=self.clip_max_v19, 
            use_spreads=False
        )
        
        fold_files_v19 = sorted([f for f in os.listdir(models_dir) if f.startswith("lag_mlp_fold") and f.endswith(".pth")])
        ckpt_v19 = torch.load(os.path.join(models_dir, fold_files_v19[0]), map_location=self.device)
        for fname in fold_files_v19:
            path = os.path.join(models_dir, fname)
            ckpt = torch.load(path, map_location=self.device)
            m = LagMLP_v19(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["output_dim"])
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.models_v19.append(m)

        # --- Load v21 (Residual) ---
        norm_v21_path = os.path.join(models_dir, "lag_mlp_v21_normalization.npz")
        self.norm_v21 = np.load(norm_v21_path)
        self.x_mean_v21 = self.norm_v21["x_mean"].astype(np.float32)
        self.x_std_v21 = self.norm_v21["x_std"].astype(np.float32)
        self.clip_min_v21 = self.norm_v21["clip_min"].astype(np.float32)
        self.clip_max_v21 = self.norm_v21["clip_max"].astype(np.float32)
        self.n_lags_v21 = int(self.norm_v21["n_lags"])
        
        self.extractor_v21 = FeatureExtractor(
            n_lags=self.n_lags_v21, 
            clip_min=self.clip_min_v21, 
            clip_max=self.clip_max_v21, 
            use_spreads=True
        )
        
        fold_files_v21 = sorted([f for f in os.listdir(models_dir) if f.startswith("lag_mlp_v21_fold") and f.endswith(".pth")])
        ckpt_v21 = torch.load(os.path.join(models_dir, fold_files_v21[0]), map_location=self.device)
        for fname in fold_files_v21:
            path = os.path.join(models_dir, fname)
            ckpt = torch.load(path, map_location=self.device)
            m = LagMLP_v21(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["output_dim"])
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.models_v21.append(m)
            
        torch.set_num_threads(1)
        self.current_seq = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if data_point.seq_ix != self.current_seq:
            self.current_seq = data_point.seq_ix
            self.extractor_v19.reset()
            self.extractor_v21.reset()

        # Stream features for both models
        # v19 (Level)
        feats_v19 = self.extractor_v19.stream(
            data_point.state, data_point.step_in_seq, data_point.seq_ix
        )
        
        # v21 (Residual)
        feats_v21 = self.extractor_v21.stream(
            data_point.state, data_point.step_in_seq, data_point.seq_ix
        )
        
        if not data_point.need_prediction:
            return None
            
        if feats_v19 is None or feats_v21 is None:
            return data_point.state.astype(np.float32) # Fallback
            
        # Inference v19
        x_v19 = (feats_v19 - self.x_mean_v19) / self.x_std_v19
        t_v19 = torch.from_numpy(x_v19).unsqueeze(0)
        pred_v19 = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            for m in self.models_v19:
                pred_v19 += m(t_v19).squeeze(0).numpy()
        pred_v19 /= len(self.models_v19)
        
        # Inference v21
        x_v21 = (feats_v21 - self.x_mean_v21) / self.x_std_v21
        t_v21 = torch.from_numpy(x_v21).unsqueeze(0)
        pred_v21_resid = np.zeros(32, dtype=np.float32)
        with torch.no_grad():
            for m in self.models_v21:
                pred_v21_resid += m(t_v21).squeeze(0).numpy()
        pred_v21_resid /= len(self.models_v21)
        pred_v21 = data_point.state.astype(np.float32) + pred_v21_resid
        
        # Blend
        # final = alpha * v19 + (1 - alpha) * v21
        final_pred = self.alphas * pred_v19 + (1.0 - self.alphas) * pred_v21
        
        return final_pred
