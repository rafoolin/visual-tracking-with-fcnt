"""Predicts heatmaps from selected VGG feature maps."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNet(nn.Module):
    """
    SNet: FCNT paper-compliant regression network that predicts a heatmap
    from selected conv4_3 feature maps.

    Architecture:
        - Conv2d: in_channels → 36, kernel_size=9, padding=4
        - ReLU
        - Conv2d: 36 → 1, kernel_size=5, padding=2

    Args:
        in_channels (int): Number of selected feature maps (default: 384)
        This default value comes from paper in the experiment section.
        weight_std: standard deviation (σ) for initializing weights.
    """

    def __init__(
        self,
        in_channels=384,
        weight_std: float = 1e-7,
    ):
        super(SNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=36,
            kernel_size=9,
            padding=4,
        )
        self.conv2 = nn.Conv2d(
            in_channels=36,
            out_channels=1,
            kernel_size=5,
            padding=2,
        )

        nn.init.normal_(self.conv1.weight, std=weight_std)
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.normal_(self.conv2.weight, std=weight_std)
        nn.init.constant_(self.conv2.bias, 0.0)

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
