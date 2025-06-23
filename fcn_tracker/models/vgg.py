"VGG16 Layer"

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG(nn.Module):
    """
    VGG16-based feature extractor for FCNT tracking.

    This wrapper splits the pretrained VGG16 model into two parts:
    - features_4: layers up to conv4_3 (inclusive)
    - features_5: layers from conv5_1 to conv5_3

    The outputs f4 and f5 correspond to feature maps after conv4_3 and conv5_3.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize the VGG feature extractor.

        Args:
            pretrained (bool): If True, loads ImageNet pretrained weights.
        """
        super(VGG, self).__init__()
        weights = VGG16_Weights.DEFAULT if pretrained else None
        vgg = vgg16(weights=weights)

        # conv1_1 to conv4_3 (index 0 to 22)
        self.features_4 = nn.Sequential(*vgg.features[:23])

        # conv5_1 to conv5_3 (index 23 to 29)
        self.features_5 = nn.Sequential(*vgg.features[23:30])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VGG network.

        Args:
            x (Tensor): Input image tensor of shape [B, 3, H, W].

        Returns:
            f4 (Tensor): Output from conv4_3, shape [B, 512, H/16, W/16].
            f5 (Tensor): Output from conv5_3, shape [B, 512, H/32, W/32].
        """
        f4 = self.features_4(x)
        f5 = self.features_5(f4)
        return f4, f5
