"""
Streaming inference using the small Micro-Mamba model trained on v19 features (window=10, residual target).
Loads:
  - models/mamba_v19_small.pth
  - models/mamba_v19_small_norm.npz (mean, std, clip_min, clip_max)
  - models/mamba_v19_small_meta.json (config)

Note: This mirrors the training setup where each sample is a single v19 feature vector
      (no multi-step unroll); the Mamba blocks run on a length-1 sequence.
"""

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from utils import DataPoint
from src.features.extractor import FeatureExtractor, feature_dim


DEVICE = torch.device("cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "mamba_v19_small.pth")
NORM_PATH = os.path.join(os.path.dirname(__file__), "models", "mamba_v19_small_norm.npz")
META_PATH = os.path.join(os.path.dirname(__file__), "models", "mamba_v19_small_meta.json")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class MambaBlock(nn.Module):
    """
    Simplified SSD-style block matching the training script:
    - Depthwise conv for local context
    - Diagonal decay per head/state
    - Gated mix of SSM output and conv output
    """

    def __init__(self, d_model: int, d_state: int, nheads: int, d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % nheads == 0, "d_model must be divisible by nheads"
        self.d_model = d_model
        self.nheads = nheads
        self.headdim = d_model // nheads
        self.d_state = d_state

        padding = d_conv - 1
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=padding,
            groups=d_model,
        )
        self.conv_activation = nn.SiLU()

        self.in_proj = nn.Linear(d_model, d_model * 3)  # value, gate, skip
        self.u_proj = nn.Linear(d_model, nheads * d_state)
        self.c_proj = nn.Linear(d_model, nheads * d_state)
        self.dt_proj = nn.Linear(d_model, nheads)

        self.A_log = nn.Parameter(torch.randn(nheads, d_state) * -0.3)
        self.dt_bias = nn.Parameter(torch.zeros(nheads))

        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        ffn_hidden = int(d_model * 2)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, x):
        B, T, D = x.shape
        x_conv = self.depthwise_conv(x.transpose(1, 2))
        x_conv = x_conv[:, :, :T]
        x_conv = self.conv_activation(x_conv).transpose(1, 2)

        proj = self.in_proj(x)
        v, gate_raw, skip = torch.split(proj, D, dim=-1)
        gate = torch.sigmoid(gate_raw)

        u = self.u_proj(x).view(B, T, self.nheads, self.d_state)
        c = self.c_proj(x).view(B, T, self.nheads, self.d_state)
        dt = torch.nn.functional.softplus(self.dt_proj(x) + self.dt_bias)

        state = x.new_zeros(B, self.nheads, self.d_state)
        A = -torch.exp(self.A_log)

        outputs = []
        for t in range(T):
            decay = torch.exp(A * dt[:, t].unsqueeze(-1))
            state = state * decay + u[:, t] * dt[:, t].unsqueeze(-1)
            h_t = (state * c[:, t]).sum(dim=-1)
            h_t = h_t.unsqueeze(-1).repeat(1, 1, self.headdim)
            h_t = h_t.reshape(B, self.d_model)
            out_t = gate[:, t] * h_t + (1 - gate[:, t]) * x_conv[:, t] + skip[:, t]
            outputs.append(out_t.unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        y = self.norm(y)
        y = y + self.dropout(self.ffn(y))
        return y


class MicroMambaModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        d_state: int,
        nheads: int,
        n_layers: int,
        d_conv: int,
        dropout: float,
        residual_output: bool = True,
    ):
        super().__init__()
        self.residual_output = residual_output
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model=d_model, d_state=d_state, nheads=nheads, d_conv=d_conv, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 32)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h)
        h_last = h[:, -1]
        out = self.head(h_last)
        return out


class PredictionModel:
    def __init__(self):
        norm = np.load(NORM_PATH)
        self.mean = norm["mean"].astype(np.float32)
        self.std = norm["std"].astype(np.float32)
        self.clip_min = norm["clip_min"].astype(np.float32)
        self.clip_max = norm["clip_max"].astype(np.float32)

        with open(META_PATH) as f:
            meta = json.load(f)

        self.n_lags = int(meta["window"])
        self.feature_dim = feature_dim(self.n_lags)
        self.extractor = FeatureExtractor(n_lags=self.n_lags, clip_min=self.clip_min, clip_max=self.clip_max)
        self.current_seq: Optional[int] = None

        self.model = MicroMambaModel(
            input_dim=self.feature_dim,
            d_model=meta["d_model"],
            d_state=meta["d_state"],
            nheads=meta["nheads"],
            n_layers=meta["layers"],
            d_conv=meta["d_conv"],
            dropout=meta["dropout"],
            residual_output=meta["residual_target"],
        ).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        seq_ix = data_point.seq_ix
        if self.current_seq is None or seq_ix != self.current_seq:
            self.extractor.reset()
            self.current_seq = seq_ix

        features = self.extractor.stream(data_point.state, data_point.step_in_seq, seq_ix=seq_ix)
        if features is None:
            return None
        if not data_point.need_prediction:
            return None

        x = ((features - self.mean) / self.std).astype(np.float32)
        x_t = torch.from_numpy(x).view(1, 1, -1).to(DEVICE)

        with torch.no_grad():
            delta = self.model(x_t).cpu().numpy().reshape(-1)
        pred = data_point.state + delta
        return pred.astype(np.float32)


if __name__ == "__main__":
    # Simple smoke test on the train file (optional)
    dp = DataPoint(seq_ix=0, step_in_seq=0, need_prediction=False, state=np.zeros(32, dtype=np.float32))
    model = PredictionModel()
    out = model.predict(dp)
    print("Init ok, prediction returned:", out)
