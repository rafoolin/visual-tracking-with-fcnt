from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from torch.nn import L1Loss
import matplotlib.pyplot as plt


import random
import cv2
import os

from PIL import Image

from entities.tracker_params import TrackerParams
from utils.roi import extract_roi

from models.vggnet import VGGNet

from models.snet import SNet
from models.gnet import GNet

import utils.heatmap as heatmap

from models.selcnn import SelCNN


from utils.feature_selection import compute_saliency

from utils.reestimate_param import reestimate_param

import utils.geo_utils as geo
import utils.visualization as viz


@dataclass
class FCNTracker:
    params: TrackerParams
    device: str

    def __post_init__(self):
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)
        self.im1_id = 0
        data_path = self.params.seq_path
        im1_path = os.path.join(data_path, f"frame_{self.im1_id:05d}.jpg")
        self.im1 = np.array(Image.open(im1_path).convert("RGB")).astype(np.float32)
        self.vgg_net = VGGNet().to(device=self.device).eval()
        self.lfeat1 = None
        self.gfeat1 = None
        self.clean_lfeat1 = None
        self.clean_gfeat1 = None
        self.map_tensor = None
        self.lid = None
        self.gid = None
        self.s_optimizer = None
        self.snet = None
        self.g_optimizer = None
        self.gnet = None
        self.preprocess = self.vgg_net.weights.transforms()

    # TODO: doc
    def initialize(self):
        """Initializer"""

        # ROI from init bounding box
        roi, *_ = extract_roi(
            im=self.im1,
            r_w_scale=self.params.s1,
            l_off=[0, 0],
            bbox=self.params.location,
            roi_size=self.params.roi_size,
        )
        roi_pil = Image.fromarray(roi.astype(np.uint8))
        # shape: [1, 3, H, W]
        roi_preprocessed = self.preprocess(roi_pil).unsqueeze(0)
        roi_tensor = roi_preprocessed.to(device=self.device)

        # VGG Net
        with torch.no_grad():
            self.lfeat1, self.gfeat1 = self.vgg_net(roi_tensor)
            # Resize global features
            self.gfeat1 = F.interpolate(
                self.gfeat1,
                size=self.lfeat1.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        # First Gaussian Map
        map1 = heatmap.get_gaussian_map(
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
        self.map_tensor = (
            torch.tensor(map1, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.conf_store = 0

        # SellCNN
        sel_snet = SelCNN(in_channels=self.lfeat1.shape[1]).to(self.device).eval()
        sel_gnet = SelCNN(in_channels=self.gfeat1.shape[1]).to(self.device).eval()

        sel_cnn_lr = float(self.params.selcnn_param.learning_rate)
        weight_decay = float(self.params.selcnn_param.weight_decay)
        momentum = float(self.params.selcnn_param.momentum)
        optimizer_s = torch.optim.SGD(
            sel_snet.parameters(),
            lr=sel_cnn_lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        optimizer_g = torch.optim.SGD(
            sel_gnet.parameters(),
            lr=sel_cnn_lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        sel_snet.train()
        sel_gnet.train()

        max_iter_select = int(self.params.max_iter_select)
        for i in range(max_iter_select):
            # Local
            out_s = sel_snet(self.lfeat1)
            loss_s = F.mse_loss(out_s, self.map_tensor)
            optimizer_s.zero_grad()
            loss_s.backward()
            optimizer_s.step()

            # Global
            out_g = sel_gnet(self.gfeat1)
            loss_g = F.mse_loss(out_g, self.map_tensor)
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            print(
                f"[SelCNN] Iter {i+1:03d}/{max_iter_select}, "
                f"Loss_s: {loss_s.item():.4f}, Loss_g: {loss_g.item():.4f}"
            )

        num_channels = int(self.params.in_channels)
        self.lid = compute_saliency(
            feature=self.lfeat1,
            map_tensor=self.map_tensor,
            model=sel_snet,
            device=self.device,
        )[:num_channels]

        self.gid = compute_saliency(
            feature=self.gfeat1,
            map_tensor=self.map_tensor,
            model=sel_gnet,
            device=self.device,
        )[:num_channels]

        self.lfeat1 = self.lfeat1[:, self.lid, :, :]
        self.gfeat1 = self.gfeat1[:, self.gid, :, :]

        # Training on SNet and GNet
        max_iter = int(self.params.max_iter)
        s_lr = float(self.params.snet_param.learning_rate)
        g_lr = float(self.params.gnet_param.learning_rate)
        self.snet = SNet(in_channels=num_channels).to(device=self.device).eval()
        self.gnet = GNet(in_channels=num_channels).to(device=self.device).eval()

        self.s_optimizer = torch.optim.SGD(
            self.snet.parameters(),
            lr=s_lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        self.g_optimizer = torch.optim.SGD(
            self.gnet.parameters(),
            lr=g_lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        # Fine-tune SNet and GNet using selected feature maps
        self.snet.train()
        self.gnet.train()

        for i in range(max_iter):
            s_pre_map = self.snet(self.lfeat1)
            g_pre_map = self.gnet(self.gfeat1)

            s_loss = F.mse_loss(s_pre_map, self.map_tensor)
            g_loss = F.mse_loss(g_pre_map, self.map_tensor)

            self.s_optimizer.zero_grad()
            s_loss.backward()
            self.s_optimizer.step()

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

    def track(self):
        pf_param = self.params.pf_param
        location = self.params.location
        im1_id = 0
        fnum = 10
        data_path = self.params.seq_path
        fea2_store = self.lfeat1.detach().clone()
        map2_store = self.map_tensor.detach().clone()

        position = np.zeros((6, fnum - im1_id + 1), dtype=np.float32)
        best_geo_param = geo.loc2affgeo(location, pf_param.p_sz)
        print("scale:", best_geo_param[2], "aspect:", best_geo_param[4])
        for im2_id in range(im1_id, fnum + 1):
            s_distractor = False
            g_distractor = False
            location_last = location.copy()
            print(f"[Tracking] Processing frame {im2_id}/{fnum}")

            im2_path = os.path.join(data_path, f"frame_{im2_id:05d}.jpg")
            im2 = Image.open(im2_path).convert("RGB")
            im2 = np.array(im2).astype(np.float64)

            # === Extract ROI ===
            roi2, roi_pos, padded_zero_map, pad = extract_roi(
                im=im2,
                l_off=[0, 0],
                r_w_scale=self.params.s2,
                bbox=location,
                roi_size=self.params.roi_size,
            )
            roi_pil = Image.fromarray(roi2.astype(np.uint8))

            # shape: [1, 3, H, W]
            roi_preprocessed = self.preprocess(roi_pil).unsqueeze(0)
            roi_tensor = roi_preprocessed.to(device=self.device)

            # === Draw particles around current best geometry ===
            geo_param = geo.draw_particles(best_geo_param, pf_param)

            # === Extract VGG features ===
            with torch.no_grad():
                lfea2, gfea2 = self.vgg_net(roi_tensor)
                # Feature map before passing into SNet/GNet
                gfea2 = F.interpolate(
                    gfea2,
                    size=lfea2.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # Select channels
            lfea2 = lfea2[:, self.lid, :, :]
            gfea2 = gfea2[:, self.gid, :, :]

            # === Compute confidence map ===
            with torch.no_grad():
                s_pre_map = self.snet(lfea2)  # shape [1, 1, H, W]
                g_pre_map = self.gnet(gfea2)  # shape [1, 1, H, W]

            # Convert to numpy, reshape [H, W]
            s_pre_map = s_pre_map.squeeze(0).squeeze(0).cpu().numpy()
            g_pre_map = g_pre_map.squeeze(0).squeeze(0).cpu().numpy()

            # Normalize
            s_pre_map = s_pre_map - s_pre_map.min()
            s_pre_map = s_pre_map / (s_pre_map.max() + 1e-8)

            g_pre_map = g_pre_map - g_pre_map.min()
            g_pre_map = g_pre_map / (g_pre_map.max() + 1e-8)

            # === Resize g_pre_map to match ROI size ===
            roi_w, roi_h = roi_pos[2], roi_pos[3]
            g_roi_map = cv2.resize(g_pre_map, (roi_w, roi_h))

            # === Place ROI map back into image-shaped padded map ===
            g_im_map = padded_zero_map.copy()

            x1, y1 = roi_pos[0], roi_pos[1]
            x2, y2 = x1 + roi_w, y1 + roi_h
            g_im_map[y1:y2, x1:x2] = g_roi_map

            # === Remove padding ===
            pad = int(pad)
            if pad > 0 and 2 * pad < min(g_im_map.shape):
                g_im_map = g_im_map[pad:-pad, pad:-pad]
            else:
                print(f"[WARN] Skipping pad crop. pad={pad}, shape={g_im_map.shape}")

            # === Threshold
            g_im_map = (g_im_map > 0.1).astype(np.float32) * g_im_map

            center = np.unravel_index(np.argmax(g_pre_map), g_pre_map.shape)
            print("[DEBUG] GNet heatmap peak location:", center)
            target_center = np.unravel_index(
                np.argmax(self.map_tensor.squeeze().cpu().numpy()), (46, 46)
            )
            print("[DEBUG] Target Gaussian peak location:", target_center)
            # === Warp the map for all particles
            wmaps = geo.warpimg(
                g_im_map,
                geo_param,
                (pf_param.p_sz, pf_param.p_sz),
            )
            # === Confidence for each particle
            g_conf = np.sum(wmaps, axis=(1, 2)) / (pf_param.p_sz**2)

            # === Ranking confidence
            scale_sq = geo_param[2, :] ** 2
            aspect = geo_param[4, :]
            g_rank_conf = g_conf * ((pf_param.p_sz**2 * scale_sq * aspect) ** 0.7)

            g_maxid = np.argmax(g_rank_conf)

            # === Resize local map to ROI size ===
            s_roi_map = cv2.resize(s_pre_map, (roi_pos[2], roi_pos[3]))

            # === Place back into padded map
            s_im_map = padded_zero_map.copy()
            x1, y1 = roi_pos[0], roi_pos[1]
            x2, y2 = x1 + roi_pos[2], y1 + roi_pos[3]
            s_im_map[y1:y2, x1:x2] = s_roi_map

            # === Remove padding
            pad = int(pad)
            if pad > 0:
                s_im_map = s_im_map[pad:-pad, pad:-pad]

            # === Threshold
            s_im_map = (s_im_map > 0.1).astype(np.float32) * s_im_map

            # === Warp
            wmaps = geo.warpimg(s_im_map, geo_param, (pf_param.p_sz, pf_param.p_sz))

            # === Confidence
            s_conf = np.sum(wmaps, axis=(1, 2)) / (pf_param.p_sz**2)

            # === Ranking
            scale_sq = geo_param[2, :] ** 2
            aspect = geo_param[4, :]
            s_rank_conf = s_conf * ((pf_param.p_sz**2 * scale_sq * aspect) ** 0.75)

            s_maxid = np.argmax(s_rank_conf)

            # Convert affine params → bbox [x, y, w, h]
            potential_location = geo.affgeo2loc(geo_param[:, g_maxid], pf_param.p_sz)
            print("potential_location", potential_location)
            # Clip box to image boundaries
            x, y, w, h = potential_location
            h_img, w_img = im2.shape[:2]
            px1 = int(np.clip(x, 0, w_img - 1))
            px2 = int(np.clip(px1 + w - 1, 0, w_img - 1))
            py1 = int(np.clip(y, 0, h_img - 1))
            py2 = int(np.clip(py1 + h - 1, 0, h_img - 1))

            # Rectify the map
            rectified_im_map = (g_im_map > 0.2).astype(np.float32) * g_im_map

            # Compute confidence inside vs. outside box
            inside_conf = np.sum(rectified_im_map[py1 : py2 + 1, px1 : px2 + 1])
            outside_conf = np.sum(rectified_im_map) - inside_conf

            # Decide if distractor
            g_distractor = outside_conf >= 0.2 * inside_conf

            # === SNet distractor check ===
            rectified_im_map = (s_im_map > 0.01).astype(np.float32) * s_im_map

            inside_conf = np.sum(rectified_im_map[py1 : py2 + 1, px1 : px2 + 1])
            outside_conf = np.sum(rectified_im_map) - inside_conf

            s_distractor = outside_conf >= 0.2 * inside_conf

            if g_distractor:  # or s_distractor
                maxconf = s_conf[s_maxid]
                maxid = s_maxid
                pre_map = s_roi_map
            else:
                maxconf = g_conf[g_maxid]
                maxid = g_maxid
                pre_map = g_roi_map

            print(
                f"[Conf] SNet max: {s_conf[s_maxid]:.4f},"
                f" GNet max: {g_conf[g_maxid]:.4f}"
            )

            if maxconf > pf_param.mv_thr:
                location = geo.affgeo2loc(geo_param[:, maxid], pf_param.p_sz)
                best_geo_param = geo_param[:, maxid].copy()
            elif s_conf[s_maxid] > pf_param.mv_thr:
                location = geo.affgeo2loc(geo_param[:, s_maxid], pf_param.p_sz)
                best_geo_param = geo_param[:, s_maxid].copy()
                maxconf = s_conf[s_maxid]

            # If confidence too low → keep previous scale/aspect (dims 3 and 5)
            if maxconf < pf_param.up_thr and (im2_id - im1_id) > 0:
                best_geo_param[2] = position[2, im2_id - im1_id - 1]  # scale
                best_geo_param[4] = position[4, im2_id - im1_id - 1]  # aspect
                location = geo.affgeo2loc(best_geo_param, pf_param.p_sz)

            # === Draw tracking result ===
            viz.drawresult(
                fno=im2_id,
                frame=im2 / 255.0,  # normalize to [0,1] (like mat2gray)
                sz=(pf_param.p_sz, pf_param.p_sz),
                mat_param=geo.affparam2mat(best_geo_param),
            )

            # === Save best_geo_param to position array ===
            position[:, im2_id - im1_id] = best_geo_param

            # === Resize saliency mask to roi_size for later use ===
            roi_size = self.params.roi_size
            mask = cv2.resize(pre_map, (roi_size, roi_size))
            mask = mask / np.max(mask) if np.max(mask) > 0 else mask
            if maxconf > self.conf_store and maxconf > pf_param.up_thr:
                # Motion offset from last location
                l_off = np.array(location_last[:2]) - np.array(location[:2])

                # Regenerate Gaussian map using new location and motion offset
                updated_map = heatmap.get_gaussian_map(
                    im_shape=im2.shape,
                    fea_shape=lfea2.shape[2:],  # [H_feat, W_feat]
                    roi_size=roi_size,
                    location=location,
                    offset=l_off,
                    scale=self.params.s2,
                    mode="gaussian",
                )

                print("lfea2", lfea2.shape)  # Expect: [1, in_channels, H, W]
                print(torch.min(lfea2), torch.max(lfea2))

                # Save for online update
                fea2_store = lfea2.detach().clone()
                map2_store = (
                    torch.tensor(updated_map, dtype=torch.float32, device=self.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                self.conf_store = maxconf

            if self.conf_store > pf_param.up_thr and (im2_id - im1_id + 1) % 20 == 0:
                self.snet.train()

                # Stack features into a batch of 2
                # [2, C, H, W]
                fea_batch = torch.cat([self.lfeat1, fea2_store], dim=0)
                # [2, 1, H, W]
                map_batch = torch.cat([self.map_tensor, map2_store], dim=0)

                # Forward
                s_pre_map = self.snet(fea_batch)

                # Loss
                s_loss = 0.5 * F.mse_loss(s_pre_map, map_batch)

                # Backward
                self.s_optimizer.zero_grad()
                s_loss.backward()
                self.s_optimizer.step()

                # Reset confidence store
                self.conf_store = pf_param.up_thr

                print(
                    f"[Update] Frame {im2_id}: "
                    f"Online update triggered, loss = {s_loss.item():.4f}"
                )

            if s_distractor and maxconf > pf_param.up_thr:
                self.snet.train()

                # Feature batch: fea2_store and lfea2
                lfea2_batch = torch.cat([fea2_store, lfea2], dim=0)  # [2, C, H, W]

                # Offset from previous position
                l_off = np.array(location_last[:2]) - np.array(location[:2])

                # Create new suppressed map
                new_map = heatmap.get_gaussian_map(
                    im_shape=im2.shape,
                    fea_shape=lfea2.shape[2:],  # (H_feat, W_feat)
                    roi_size=roi_size,
                    location=location,
                    offset=l_off,
                    scale=self.params.s2,
                    mode="gaussian",
                )
                
                new_map_tensor = (
                    torch.tensor(new_map, dtype=torch.float32, device=self.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                # Expand to batch
                map_batch = torch.cat([map2_store, new_map_tensor], dim=0)

                # Generate mask for suppression (only where map <= 0)
                suppression_mask = (new_map_tensor <= 0).float()

                # Iterate update
                for _ in range(10):
                    s_out = self.snet(lfea2_batch)  # [2, 1, H, W]

                    loss1 = 0.5 * F.mse_loss(s_out[0:1], map2_store)
                    loss2 = 0.5 * F.mse_loss(
                        s_out[1:2] * suppression_mask, new_map_tensor * suppression_mask
                    )

                    loss = loss1 + loss2
                    self.s_optimizer.zero_grad()
                    loss.backward()
                    self.s_optimizer.step()

                print(
                    f"[Distractor Update] Frame {im2_id}: "
                    f"SNet updated (loss1={loss1.item():.4f}, loss2={loss2.item():.4f})"
                )

                if maxconf < pf_param.minconf:
                    pf_param.minconf = maxconf

                if im2_id == self.params.check_num:
                    pf_param = reestimate_param(pf_param)

            print("Raw SNet:", s_pre_map.min(), s_pre_map.max())
            print("Raw GNet:", g_pre_map.min(), g_pre_map.max())
