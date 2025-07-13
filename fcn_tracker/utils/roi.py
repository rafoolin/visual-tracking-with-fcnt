import numpy as np
import cv2


def extract_roi(im, bbox, l_off, roi_size, r_w_scale):
    """
    Args:
        im:      input image as np.array (H, W, 3)
        bbox:    [x, y, w, h] format
        l_off:   [dx, dy] offset for position adjustment
        roi_size: final size (int, e.g. 368)
        r_w_scale: [sx, sy] scaling factors for ROI window

    Returns:
        roi: cropped and resized patch (roi_size x roi_size x 3)
        roi_pos: [x1, y1, w, h] in original image coords
        preim: blank image same size as padded image
        pad: number of pixels padded
    """

    h, w, _ = im.shape
    win_w, win_h = bbox[2], bbox[3]
    win_lt_x, win_lt_y = bbox[0], bbox[1]

    win_cx = round(win_lt_x + win_w / 2 + l_off[0])
    win_cy = round(win_lt_y + win_h / 2 + l_off[1])

    roi_w = r_w_scale[0] * win_w
    roi_h = r_w_scale[1] * win_h

    x1 = int(round(win_cx - roi_w / 2))
    y1 = int(round(win_cy - roi_h / 2))
    x2 = int(round(win_cx + roi_w / 2))
    y2 = int(round(win_cy + roi_h / 2))

    pad = 0
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        pad = int(max(-x1, -y1, x2 - w, y2 - h, 0)) + 1
        im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
        x1 += pad
        x2 += pad
        y1 += pad
        y2 += pad

    roi = im[y1:y2, x1:x2]
    roi = cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)

    preim = np.zeros((im.shape[0], im.shape[1]))
    roi_pos = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

    return roi, roi_pos, preim, pad
