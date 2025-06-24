"""
Selection Convolutional Neural Network (SelCNN) for scoring feature maps.
"""

import torch
import torch.nn as nn


class SelCNN(nn.Module):
    """
    To select feature map in FCNT.
    Predicts a 2D heatmap from VGG feature maps using a simple regression model.
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 1,
        dropout_rate: float = 0.3,
        kernel_size: int = 3,
        padding: int = 1,
        weight_std: float = 1e-7,
        bias_value: float = 0.0,
    ):

        super(SelCNN, self).__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        nn.init.normal_(self.conv.weight, mean=0.0, std=weight_std)
        nn.init.constant_(self.conv.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SelCNN.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            Tensor: Output saliency map of shape (B, 1, H, W)
        """
        x = self.dropout(x)
        return self.conv(x)
