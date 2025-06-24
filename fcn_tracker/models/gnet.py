"""Predicts heatmaps from selected VGG feature maps."""

import torch
import torch.nn.functional as F
import torch.nn as nn


class GNet(nn.Module):
    """
    GNet: A lightweight convolutional network that transforms selected feature maps
    into a single-channel heatmap for object localization.

    Architecture:
        - Conv2d: in_channels → 36, kernel_size=9, padding=4
        - ReLU
        - Conv2d: 36 → 1, kernel_size=5, padding=2

    Args:
        in_channels (int): Number of input feature map channels (default: 384).
    """

    def __init__(self, in_channels: int = 384):
        super(GNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=36,
            kernel_size=5,
            padding=4,
        )
        self.conv3 = nn.Conv2d(
            in_channels=36,
            out_channels=1,
            kernel_size=5,
            padding=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to produce heatmap.

        Args:
            x (Tensor): Input feature map tensor of shape [B, in_channels, H, W]

        Returns:
            Tensor: Predicted heatmap of shape [B, 1, H, W]
        """
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x
