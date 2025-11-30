import torch
import torch.nn as nn
import torch.nn.functional as F
from .torchtsmixer.tsmixer import TSMixer

class RevINLite(nn.Module):
    """
    Robust Reversible Instance Normalization for short windows.
    Blends per-instance stats with global training stats to prevent instability.
    
    x_norm = (x - mu_eff) / sigma_eff
    mu_eff = alpha * mu_window + (1-alpha) * mu_global
    """
    def __init__(self, num_features, global_mean, global_std, alpha=0.5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.alpha = alpha
        self.affine = affine
        
        # Register global stats as buffers (non-trainable)
        self.register_buffer('global_mean', torch.tensor(global_mean).float())
        self.register_buffer('global_std', torch.tensor(global_std).float())
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
            
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            # x: [B, L, C]
            # Compute window stats
            self.mean_window = x.mean(dim=1, keepdim=True).detach()
            self.std_window = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            
            # Blend stats
            # Broadcast global stats to match batch
            self.mean_eff = self.alpha * self.mean_window + (1 - self.alpha) * self.global_mean
            self.std_eff = self.alpha * self.std_window + (1 - self.alpha) * self.global_std
            
            x = (x - self.mean_eff) / self.std_eff
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
            
        elif mode == 'denorm':
            # Slice stats first if output dim < input dim
            out_dim = x.shape[-1]
            mean_eff = self.mean_eff[..., :out_dim]
            std_eff = self.std_eff[..., :out_dim]
            
            if self.affine:
                # Slice affine params to match output dim
                bias = self.affine_bias[:out_dim]
                weight = self.affine_weight[:out_dim]
                x = (x - bias) / (weight + 1e-5)
            
            x = x * std_eff + mean_eff
            return x

class TSMixerRefined(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 seq_len, 
                 pred_len,
                 global_mean, 
                 global_std,
                 d_model=64, 
                 num_blocks=4, 
                 dropout=0.1,
                 alpha=0.5):
        super().__init__()
        
        # 1. Robust Normalization
        self.revin = RevINLite(input_channels, global_mean, global_std, alpha=alpha)
        
        # 2. Input Stem (Compression)
        # Projects high-dim hybrid features (192) to dense d_model (64)
        self.stem = nn.Linear(input_channels, d_model)
        self.stem_act = nn.GELU()
        
        # 3. TSMixer Backbone
        # Note: input_channels to mixer is now d_model
        # We want output of mixer to be d_model, then project to output_channels
        self.backbone = TSMixer(
            sequence_length=seq_len,
            prediction_length=pred_len,
            input_channels=d_model,
            output_channels=d_model, # Keep latent dim
            num_blocks=num_blocks,
            ff_dim=d_model * 2,
            dropout_rate=dropout,
            activation_fn="gelu", # Modern activation
            normalize_before=True,
            norm_type="batch"
        )
        
        # 4. Output Head
        self.head = nn.Linear(d_model, output_channels)
        
    def forward(self, x):
        # x: [B, L, 192]
        
        # Norm
        x = self.revin(x, 'norm')
        
        # Stem: [B, L, 192] -> [B, L, d_model]
        x = self.stem(x)
        x = self.stem_act(x)
        
        # Mixer: [B, L, d_model] -> [B, T, d_model]
        x = self.backbone(x)
        
        # Head: [B, T, d_model] -> [B, T, 32]
        x = self.head(x)
        
        # Denorm
        x = self.revin(x, 'denorm')
        
        return x
