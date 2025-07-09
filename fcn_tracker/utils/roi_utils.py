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

    x1 = int(win_cx - round(roi_w / 2))
    y1 = int(win_cy - round(roi_h / 2))
    x2 = int(win_cx + round(roi_w / 2))
    y2 = int(win_cy + round(roi_h / 2))

    clip = min([x1, y1, h - y2, w - x2])
    pad = 0
    if clip <= 0:
        pad = abs(clip) + 1
        im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
        x1 += pad
        x2 += pad
        y1 += pad
        y2 += pad

    cropped = im[y1 : y2 + 1, x1 : x2 + 1, :]
    roi = cv2.resize(cropped, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)

    preim = np.zeros((im.shape[0], im.shape[1]))
    roi_pos = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

    return roi, roi_pos, preim, pad


def preprocess_roi(im):
    """
    Args:
        im: np.array of shape (H, W, 3) in RGB, float32

    Returns:
        I: np.array of shape (W, H, 3) in BGR with mean subtracted
    """
    if not isinstance(im, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(im)}")
    mean_pix = [103.939, 116.779, 123.68]  # BGR

    # Transpose HxW → WxH (same as permute([2,1,3]) in MATLAB)
    im = np.transpose(im, (1, 0, 2))  # (W, H, C)

    # Convert RGB to BGR
    im = im[:, :, ::-1]

    # Subtract mean pixel values
    I = np.empty_like(im)
    I[:, :, 0] = im[:, :, 0] - mean_pix[0]  # B
    I[:, :, 1] = im[:, :, 1] - mean_pix[1]  # G
    I[:, :, 2] = im[:, :, 2] - mean_pix[2]  # R

    return I
