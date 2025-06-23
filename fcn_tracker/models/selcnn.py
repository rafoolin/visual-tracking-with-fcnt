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

    def __init__(self, in_channels: int = 512, dropout: float = 0.5):
        """
        Args:
            in_channels (int): Number of feature maps from VGG layer (e.g., 512).
            dropout (float): Dropout rate.
        """
        super(SelCNN, self).__init__()
        self.dropout = nn.Dropout2d(p=dropout)
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SelCNN.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            Tensor: Output saliency map of shape (B, 1, H, W)
        """
        x = self.dropout(x)
        return self.conv1x1(x)
