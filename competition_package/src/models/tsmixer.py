import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeMixing(nn.Module):
    def __init__(self, seq_len, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch, seq_len, channels]
        # Mix over time: apply to each channel independently
        # Transpose to [batch, channels, seq_len]
        x = x.transpose(1, 2)
        x = self.net(x)
        # Transpose back to [batch, seq_len, channels]
        return x.transpose(1, 2)

class FeatureMixing(nn.Module):
    def __init__(self, input_channels, hidden_channels, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, input_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch, seq_len, channels]
        # Mix over features: apply to each time step independently
        return self.net(x)

class MixerLayer(nn.Module):
    def __init__(self, seq_len, channels, hidden_channels, dropout):
        super().__init__()
        # Normalization applied to both dims: [batch, seq_len, channels]
        self.norm_time = nn.LayerNorm([seq_len, channels])
        self.time_mixing = TimeMixing(seq_len, dropout)
        
        self.norm_feat = nn.LayerNorm([seq_len, channels])
        self.feature_mixing = FeatureMixing(channels, hidden_channels, dropout)

    def forward(self, x):
        # Time Mixing Block with Residual
        # Paper: Norm -> TimeMix -> Add
        x_norm = self.norm_time(x)
        x = x + self.time_mixing(x_norm)
        
        # Feature Mixing Block with Residual
        # Paper: Norm -> FeatMix -> Add
        x_norm = self.norm_feat(x)
        x = x + self.feature_mixing(x_norm)
        
        return x

class TSMixer(nn.Module):
    def __init__(self, 
                 sequence_length, 
                 prediction_length, 
                 input_channels, 
                 output_channels, 
                 num_blocks=4, 
                 hidden_channels=64, 
                 dropout=0.1):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Input projection if needed (optional, usually raw input is fine if channels match)
        # In this specific problem, input_channels == output_channels == 32
        
        self.mixer_blocks = nn.ModuleList([
            MixerLayer(sequence_length, input_channels, hidden_channels, dropout)
            for _ in range(num_blocks)
        ])
        
        # Temporal Projection: Map sequence_length -> prediction_length
        # Applied to the time dimension
        self.temporal_projection = nn.Linear(sequence_length, prediction_length)
        
        # Output projection if input_channels != output_channels
        if input_channels != output_channels:
            self.output_projection = nn.Linear(input_channels, output_channels)
        else:
            self.output_projection = nn.Identity()

    def forward(self, x):
        # x: [batch, sequence_length, input_channels]
        
        for block in self.mixer_blocks:
            x = block(x)
            
        # Temporal Projection: [batch, seq_len, channels] -> [batch, pred_len, channels]
        # Transpose to apply linear layer to time dim
        x = x.transpose(1, 2) # [batch, channels, seq_len]
        x = self.temporal_projection(x)
        x = x.transpose(1, 2) # [batch, pred_len, channels]
        
        # Output Projection (if needed)
        x = self.output_projection(x)
        
        return x

if __name__ == '__main__':
    # Quick smoke test
    batch_size = 32
    seq_len = 10
    pred_len = 1
    channels = 32
    
    model = TSMixer(seq_len, pred_len, channels, channels)
    x = torch.randn(batch_size, seq_len, channels)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (batch_size, pred_len, channels)
    print("Smoke test passed!")
