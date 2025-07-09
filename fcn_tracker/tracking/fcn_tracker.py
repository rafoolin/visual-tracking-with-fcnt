from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import L1Loss


import random
import cv2
import os

from PIL import Image

from entities.tracker_params import TrackerParams
from utils.roi_utils import extract_roi, preprocess_roi

from models.vggnet import VGGNet

from utils.heatmap import get_gaussian_map

from models.selcnn import SelCNN


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

        # ROI from init bounding box
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

        # VGG Net
        vgg_net = VGGNet().to(device=self.device).eval()
        with torch.no_grad():
            self.lfeat1, self.gfeat1 = vgg_net(roi_tensor)

        # Resize global features
        self.gfeat1 = F.interpolate(
            self.gfeat1,
            size=self.lfeat1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # First Gaussian Map
        self.map1 = get_gaussian_map(
            im_shape=self.im1.shape,
            # torch: [B, C, H_feat, W_feat]
            fea_shape=self.lfeat1.shape[2:],
            roi_size=self.params.roi_size,
            location=self.params.location,
            offset=[0, 0],
            scale=self.params.s1,
            mode="gaussian",
        )
        # [1, C, H, W]
        map_tensor = (
            torch.tensor(self.map1, dtype=torch.float32, device=self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        self.conf_store = 0

        # SellCNN
        sel_snet = SelCNN(in_channels=self.lfeat1.shape[1]).to(self.device)
        sel_gnet = SelCNN(in_channels=self.gfeat1.shape[1]).to(self.device)

        lr = float(self.params.selcnn_param.learning_rate)
        optimizer_s = torch.optim.SGD(sel_snet.parameters(), lr=lr)
        optimizer_g = torch.optim.SGD(sel_gnet.parameters(), lr=lr)

        sel_snet.train()
        sel_gnet.train()

        max_iter_select = int(self.params.max_iter_select)
        for i in range(max_iter_select):
            # Local
            out_s = sel_snet(self.lfeat1)
            loss_s = F.mse_loss(out_s, map_tensor)
            optimizer_s.zero_grad()
            loss_s.backward()
            optimizer_s.step()

            # Global
            out_g = sel_gnet(self.gfeat1)
            loss_g = F.mse_loss(out_g, map_tensor)
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            print(
                f"[SelCNN] Iter {i+1:03d}/{max_iter_select}, "
                f"Loss_s: {loss_s.item():.4f}, Loss_g: {loss_g.item():.4f}"
            )
