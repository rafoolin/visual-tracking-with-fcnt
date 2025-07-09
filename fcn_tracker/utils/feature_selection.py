import torch
import numpy as np


def compute_saliency(
    feature: torch.Tensor,
    device: str,
    map_tensor: torch.Tensor,
    model: torch.nn.Module,
) -> list[int]:
    """
    Returns: list of most salient channel indices
    """
    model.to(device=device).eval()
    feature = feature.clone().detach().requires_grad_(True)

    out = model(feature)
    loss_map = (out - map_tensor).pow(2).sum()
    loss_map.backward(retain_graph=True)

    grad_1 = feature.grad.detach().clone()

    # Backward2 approximation: gradients w.r.t. constant 1
    dummy = torch.ones_like(out, requires_grad=True)
    grad_2 = torch.autograd.grad(out, feature, grad_outputs=dummy, retain_graph=True)[
        0
    ].detach()

    # Saliency formula
    # sum over H, W
    saliency = -torch.sum(grad_1 * feature + 0.5 * grad_2 * feature**2, dim=(2, 3))
    # [C]
    saliency = saliency.squeeze(0).cpu().detach().numpy()

    # descending
    return list(np.argsort(saliency)[::-1])
