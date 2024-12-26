import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()
        self.np = sum(x.numel() for x in self.parameters())  # number params
        self.type = 'ChannelAttention'  # module type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input."""
        return x * self.act(self.fc(self.pool(x)))

class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel_size."""
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.act = nn.Sigmoid()
        self.np = sum(x.numel() for x in self.parameters())  # number params
        self.type = 'SpatialAttention'  # module type

    def forward(self, x):
        """Forward pass of spatial attention."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.act(self.conv(scale))
        return x * scale

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction_ratio=16):
        """Initialize CBAM with channel and spatial attention."""
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        self.np = sum(x.numel() for x in self.parameters())  # number params
        self.type = 'CBAM'  # module type

    def forward(self, x):
        """Forward pass of CBAM."""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x 