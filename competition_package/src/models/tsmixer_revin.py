import torch
import torch.nn as nn
from .torchtsmixer import TSMixer
from .revin import RevIN

class TSMixerRevIN(nn.Module):
    """
    Wraps TSMixer with Reversible Instance Normalization (RevIN) to handle non-stationarity.
    Supports different input/output channel counts by slicing RevIN stats during denormalization.
    """
    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_channels: int,
        output_channels: int = None,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        ff_dim: int = 64,
        normalize_before: bool = True,
        norm_type: str = "batch",
        revin_affine: bool = True
    ):
        super().__init__()
        self.revin = RevIN(input_channels, affine=revin_affine)
        self.output_channels = output_channels if output_channels is not None else input_channels
        
        self.backbone = TSMixer(
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            input_channels=input_channels,
            output_channels=self.output_channels,
            activation_fn=activation_fn,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            normalize_before=normalize_before,
            norm_type=norm_type
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]
        
        # 1. Normalize input (statistics computed on all input channels)
        x = self.revin(x, 'norm')
        
        # 2. Backbone forward pass
        # TSMixer handles the projection from input_channels -> output_channels
        x = self.backbone(x)
        
        # 3. Denormalize output
        # If output channels < input channels, we assume the first K channels are the target.
        # RevIN stats are [Batch, 1, Input_Channels].
        # We need to apply only the subset corresponding to Output_Channels.
        
        if x.shape[-1] != self.revin.mean.shape[-1]:
            # Slice stats for denormalization
            # Assuming Affine parameters also need slicing if they exist
            
            # Standardize -> Scale -> Shift
            # x = (x - affine_bias) / affine_weight  <-- skipped, we assume RevIN handles this internally?
            # Wait, my RevIN implementation has _denormalize method.
            # I cannot easily call _denormalize with sliced stats because it uses self.mean/std internally.
            
            # Solution: Manually denormalize here using exposed stats
            
            stdev = self.revin.stdev[:, :, :self.output_channels]
            mean = self.revin.mean[:, :, :self.output_channels]
            
            # Handle Affine if enabled
            if self.revin.affine:
                weight = self.revin.affine_weight[:self.output_channels]
                bias = self.revin.affine_bias[:self.output_channels]
                x = x - bias
                x = x / (weight + self.revin.eps*0)
            
            x = x * stdev
            x = x + mean
            return x
        else:
            # Standard case: dims match
            return self.revin(x, 'denorm')