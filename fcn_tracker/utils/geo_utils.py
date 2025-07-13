import torch.nn.functional as F
import torch

import numpy as np


def affgeo2loc(geo_param, p_sz):
    """
    Convert affine geometric params to bounding box location.

    Args:
        geo_param (np.ndarray): shape (6,) = [cx, cy, scale, _, aspect, _]
        p_sz (int): particle size reference

    Returns:
        list: [x, y, w, h], rounded integer coordinates
    """
    cx, cy, scale, _, aspect, _ = geo_param
    w = scale * p_sz
    h = w * aspect
    tlx = cx - (w - 1) / 2
    tly = cy - (h - 1) / 2
    return [int(round(tlx)), int(round(tly)), int(round(w)), int(round(h))]


def draw_particles(geo_param, pf_param):
    """
    Sample particle hypotheses around a center affine parameter.

    Args:
        geo_param (np.ndarray): shape (6,), current best affine param
        pf_param (object): has fields `p_num` (int) and `affsig` (List[float] of len 6)

    Returns:
        np.ndarray: shape (6, p_num), sampled affine parameters
    """
    geo_param = np.tile(geo_param[:, np.newaxis], (1, pf_param.p_num))  # shape (6, N)
    noise = (
        np.random.randn(6, pf_param.p_num) * np.array(pf_param.affsig)[:, np.newaxis]
    )
    geo_params = geo_param + noise
    return geo_params


import numpy as np


def affparam2mat(p):
    """
    Converts affine parameters to 2x3 transformation matrices.

    Args:
        p (np.ndarray): Affine params of shape (6,) or (6, N),
            representing [dx, dy, scale, theta, aspect, skew].

    Returns:
        np.ndarray: 2x3 affine matrices for each column of p.
                    Shape is (2, 3, N) or (2, 3) if N=1.
    """
    p = np.asarray(p, dtype=np.float32)
    # if np.any(p[2, :] == 0) or np.any(p[4, :] == 0):
    #     raise ValueError("Invalid affine parameters: scale or aspect is zero")
    if p.ndim == 1:
        if p.size != 6:
            raise ValueError("Expected 6 parameters for a single affine transform.")
        p = p.reshape(6, 1)
    elif p.shape[0] != 6:
        raise ValueError(f"Expected shape (6, N), got {p.shape}")

    dx, dy, s, th, r, phi = p[0], p[1], p[2], p[3], p[4], p[5]

    cth = np.cos(th)
    sth = np.sin(th)
    cph = np.cos(phi)
    sph = np.sin(phi)

    ccc = cth * cph**2
    ccs = cth * cph * sph
    css = cth * sph**2
    scc = sth * cph**2
    scs = sth * cph * sph
    sss = sth * sph**2

    q = np.zeros((2, 3, p.shape[1]), dtype=np.float32)

    q[0, 0, :] = s * (ccc + scs + r * (css - scs))
    q[0, 1, :] = s * (r * (ccs - scc) - ccs - sss)
    q[0, 2, :] = dx

    q[1, 0, :] = s * (scc - ccs + r * (ccs + sss))
    q[1, 1, :] = s * (r * (ccc + scs) - scs + css)
    q[1, 2, :] = dy

    return q[:, :, 0] if q.shape[2] == 1 else q


def loc2affgeo(location, p_sz):
    """
    Convert a bounding box to affine geometry parameters.

    Args:
        location (list or array): [x, y, w, h]
        p_sz (float): particle template size

    Returns:
        np.ndarray: shape (6,), affine geometry parameters
    """
    x, y, w, h = location
    cx = x + (w - 1) / 2
    cy = y + (h - 1) / 2
    scale = w / p_sz
    aspect = h / w
    return np.array([cx, cy, scale, 0.0, aspect, 0.0], dtype=np.float32)



def warpimg(img, p, sz):
    """
    Warp a single-channel image using N affine transformations.

    Args:
        img (np.ndarray): (H_img, W_img) input image
        p (np.ndarray): (6, N) affine parameters
        sz (tuple): output size (H_out, W_out)

    Returns:
        np.ndarray: (N, H_out, W_out) warped images
    """
    H_out, W_out = sz
    H_img, W_img = img.shape
    N = p.shape[1]

    # Convert image to tensor [1, 1, H, W] → [N, 1, H, W]
    img_tensor = (
        torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    )  # [1, 1, H, W]
    img_tensor = img_tensor.expand(N, -1, -1, -1)

    # Affine matrices in pixel coords → convert to normalized for grid_sample
    p_mat = affparam2mat(p)  # shape: (2, 3, N)
    theta = torch.from_numpy(p_mat).float().permute(2, 0, 1)  # (N, 2, 3)

    # Normalize the translation offsets for grid_sample
    theta[:, 0, 2] = theta[:, 0, 2] / (W_img / 2)
    theta[:, 1, 2] = theta[:, 1, 2] / (H_img / 2)

    # Generate sampling grid
    grid = F.affine_grid(theta, size=[N, 1, H_out, W_out], align_corners=False)

    # Sample
    warped = F.grid_sample(
        img_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    return warped.squeeze(1).cpu().numpy()