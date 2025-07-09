from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import random
import cv2
import os

from PIL import Image

from entities.tracker_params import TrackerParams
from utils.roi_utils import extract_roi, preprocess_roi

from models.vggnet import VGGNet


@dataclass
class FCNTracker:
    config: any
    params: TrackerParams
    device: str

    def __post_init__(self):
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

        self.im1_id = 0
        data_path = self.config["dataset"]["seq_path"]
        self.im1_path = os.path.join(data_path, f"frame_{self.im1_id:05d}.jpg")
        self.im1 = cv2.imread(self.im1_path).astype(np.float32)
        # Convert grayscale to RGB
        if self.im1.ndim != 3 or self.im1.shape[2] != 3:
            self.im1 = np.stack([self.im1[:, :, 0]] * 3, axis=-1, dtype=np.float32)

    def initialize(self):
        roi, roi_pos, preim, pad = extract_roi(
            self.im1,
            self.params.location,
            [0, 0],
            self.params.roi_size,
            self.params.s1,
        )

        roi_preprocessed = preprocess_roi(roi)
        # shape: [1, 3, H, W]
        roi_tensor = (
            torch.from_numpy(roi_preprocessed)
            .to(device=self.device)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        vgg_net = VGGNet().to(device=self.device).eval()
        with torch.no_grad():
            self.lfeat1, self.gfeat1 = vgg_net(roi_tensor)

        # 4. Resize global features
        gfea1_resized = F.interpolate(
            self.gfeat1,
            size=self.lfeat1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        
