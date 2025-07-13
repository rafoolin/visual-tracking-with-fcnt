import numpy as np
import cv2

from utils.roi import extract_roi


def get_gaussian_map(
    im_shape,
    fea_shape,
    roi_size,
    location,
    offset,
    scale,
    mode="gaussian",
):
    h_img, w_img = im_shape[:2]
    map_full = np.zeros((h_img, w_img), dtype=np.float32)

    if mode == "box":
        x, y, w, h = location
        map_full[y : y + h, x : x + w] = 1.0

    elif mode == "gaussian":
        w_box, h_box = location[2], location[3]
        scale_val = min(w_box, h_box) / 3

        # Create square Gaussian and resize to bbox size
        square_size = int(min(w_box, h_box))
        gauss = cv2.getGaussianKernel(square_size, scale_val)
        mask = gauss @ gauss.T
        mask = cv2.resize(mask, (w_box, h_box))  # Now match object shape
        mask = mask / np.max(mask)

        x1, y1 = int(location[0]), int(location[1])
        x2, y2 = x1 + w_box, y1 + h_box

        # Padding if out of bounds
        pad = 0
        if min(x1, y1, h_img - y2, w_img - x2) < 0:
            pad = abs(min(x1, y1, h_img - y2, w_img - x2)) + 1
            map_full = np.pad(map_full, ((pad, pad), (pad, pad)))
            x1 += pad
            y1 += pad
            x2 += pad
            y2 += pad

        map_full[y1:y2, x1:x2] = mask

        if pad > 0:
            map_full = map_full[pad:-pad, pad:-pad]

    else:
        raise ValueError("Unknown map type")

    # Make 3-channel for extract_roi compatibility
    map_full = np.repeat(map_full[:, :, np.newaxis], 3, axis=2)

    roi_map, *_ = extract_roi(map_full, location, offset, roi_size, scale)

    map_resized = cv2.resize(roi_map[:, :, 0], (fea_shape[1], fea_shape[0]))

    return map_resized
