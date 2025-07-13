import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def drawbox(sz, param, color="r", linewidth=2.5):
    """
    Draw an affine-transformed box using matplotlib.

    Args:
        sz (tuple): (width, height)
        param (np.ndarray): either 2x3 affine matrix or 6D parameter vector
        color (str): box color
        linewidth (float): line width
    """
    w, h = sz

    # Check if param is a 6D vector → convert to 2x3 affine matrix
    if isinstance(param, (list, tuple, np.ndarray)) and len(param) == 6:
        p = np.asarray(param)
        M = np.array(
            [
                [p[0], p[2], p[3]],
                [p[1], p[4], p[5]],
            ]
        )  # 2x3 affine matrix
    elif isinstance(param, np.ndarray) and param.shape == (2, 3):
        M = param
    else:
        raise ValueError("Invalid affine parameters")

    # Define corners centered at origin (same order as MATLAB)
    corners = np.array(
        [
            [-w / 2, -h / 2],
            [w / 2, -h / 2],
            [w / 2, h / 2],
            [-w / 2, h / 2],
            [-w / 2, -h / 2],  # Close loop
        ]
    )  # shape (5, 2)

    # Convert to homogeneous and transform
    ones = np.ones((corners.shape[0], 1))
    corners_h = np.hstack([corners, ones])  # shape (5, 3)
    transformed = (M @ corners_h.T).T  # shape (5, 2)

    # Plot box
    plt.plot(transformed[:, 0], transformed[:, 1], color=color, linewidth=linewidth)

    # Plot center dot
    center = np.mean(transformed[:4, :], axis=0)
    plt.plot(center[0], center[1], color=color, marker="o")


def drawresult(fno, frame, sz, mat_param):
    """
    Display the tracking result for a single frame.

    Args:
        fno (int): Frame number.
        frame (np.ndarray): Image to display (H, W, 3), RGB or BGR.
        sz (tuple): Target size (width, height) for the box.
        mat_param (np.ndarray): 2x3 affine matrix.
    """
    # Convert to RGB for matplotlib if needed
    if frame.shape[2] == 3 and np.max(frame) > 1.0:
        frame = frame / 255.0
    frame_rgb = frame[..., ::-1]  # BGR to RGB

    plt.figure(1, figsize=(8, 6))
    plt.clf()
    plt.imshow(frame_rgb, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.title(f"Frame {fno}", fontsize=14, color="yellow")

    # Draw the box
    drawbox(sz, mat_param)

    plt.pause(0.001)


def save_heatmap(input_data, save_path: str, colormap="jet", resize_to=None):
    """
    Save a heatmap from tensor, array, or PIL image, with optional resizing and colormap.

    Args:
        input_data: torch.Tensor [1,C,H,W] or np.ndarray [C,H,W] or PIL.Image.Image
        save_path (str): Path to save the heatmap
        colormap (str): matplotlib colormap name (default: 'jet')
        resize_to (tuple): Optional (width, height) to resize the heatmap
    """
    # Convert to numpy array in shape [H, W]
    if isinstance(input_data, torch.Tensor):
        array = input_data.detach().cpu().numpy()
        if array.ndim == 4:
            array = array.squeeze(0)
        if array.ndim == 3:
            array = np.mean(array, axis=0)
        elif array.ndim == 2:
            pass
        else:
            raise ValueError("Unsupported tensor shape")
    elif isinstance(input_data, np.ndarray):
        array = input_data
        if array.ndim == 3:
            array = np.mean(array, axis=0)
        elif array.ndim != 2:
            raise ValueError("Unsupported numpy array shape")
    elif isinstance(input_data, Image.Image):
        array = np.array(input_data)
        if array.ndim == 3:
            array = np.mean(array, axis=2)
    else:
        raise TypeError("Unsupported input type")

    # Normalize
    array = array.astype(np.float32)
    array -= array.min()
    array /= array.max() + 1e-8

    # Apply colormap
    heatmap_color = cm.get_cmap(colormap)(array)[:, :, :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    # Convert to PIL and resize
    heatmap_image = Image.fromarray(heatmap_color)
    if resize_to:
        heatmap_image = heatmap_image.resize(resize_to, Image.BILINEAR)

    heatmap_image.save(save_path)


def visualize_image(img, title: str = "Image", figsize=(5, 5)):
    """
    Display an image (tensor, NumPy array, or PIL) using matplotlib.

    Args:
        img: input image (torch.Tensor, np.ndarray, or PIL.Image)
        title (str): title of the plot window
        figsize (tuple): size of the figure
    """
    # Convert torch.Tensor to NumPy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.shape[0] == 3:  # CHW -> HWC
            img = img.permute(1, 2, 0)
        img = img.numpy()

    # Convert PIL Image to NumPy
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Ensure dtype is uint8 for display
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Display the image
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()
