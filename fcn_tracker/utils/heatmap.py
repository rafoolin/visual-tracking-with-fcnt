import numpy as np
import cv2

from utils.roi_utils import extract_roi


def get_gaussian_map(
    im_shape,
    fea_shape,
    roi_size,
    location,
    offset,
    scale,
    mode="gaussian",
):
    """
    Replicates MATLAB's GetMap function in Python.

    Args:
        im_shape (tuple): (H, W) of the input image
        fea_shape (tuple): (H_feat, W_feat) of the feature map
        roi_size (int): output ROI size (typically square)
        location (list): [x, y, w, h] — bounding box
        offset (list): [dx, dy]
        scale (list): [scale_w, scale_h]
        mode (str): 'gaussian' or 'box'

    Returns:
        numpy.ndarray: map of shape (H_feat, W_feat), single channel
    """
    h_img, w_img = im_shape[:2]
    map_full = np.zeros((h_img, w_img), dtype=np.float32)

    if mode == "box":
        x, y, w, h = location
        map_full[y : y + h, x : x + w] = 1.0

    elif mode == "gaussian":
        w_box, h_box = location[2], location[3]
        scale_val = min(w_box, h_box) / 3
        gaussian_shape = (int(min(w_box, h_box)), int(min(w_box, h_box)))

        # Create 2D Gaussian kernel
        gauss = cv2.getGaussianKernel(gaussian_shape[0], scale_val)
        mask = gauss @ gauss.T
        mask = cv2.resize(mask, (int(w_box), int(h_box)))
        mask = mask / np.max(mask)
        x1, y1 = location[0], location[1]
        x2, y2 = x1 + w_box, y1 + h_box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Handle out-of-bounds by padding
        pad_x1, pad_y1 = max(0, -x1), max(0, -y1)
        pad_x2, pad_y2 = max(0, x2 - w_img), max(0, y2 - h_img)

        map_full = np.pad(map_full, ((pad_y1, pad_y2), (pad_x1, pad_x2)))
        x1 += pad_x1
        y1 += pad_y1
        x2 += pad_x1
        y2 += pad_y1

        map_full[y1:y2, x1:x2] = mask
        # Remove padding if added
        if pad_y1 or pad_y2 or pad_x1 or pad_x2:
            map_full = map_full[pad_y1 : h_img + pad_y1, pad_x1 : w_img + pad_x1]

    else:
        raise ValueError("Unknown map type")

    # Ensure 3D shape for compatibility with extract_roi
    if map_full.ndim == 2:
        map_full = np.repeat(map_full[:, :, np.newaxis], 3, axis=2)
    # Extract ROI
    roi_map, *_ = extract_roi(map_full, location, offset, roi_size, scale)
    # Resize to match feature map and return only 2D map
    map_resized = cv2.resize(roi_map[:, :, 0], (fea_shape[1], fea_shape[0]))

    return map_resized
