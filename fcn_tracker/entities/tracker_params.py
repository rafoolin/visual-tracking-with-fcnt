from dataclasses import dataclass, field
from typing import List

from entities.particle_filter_params import ParticleFilterParams


@dataclass
class SelCNNParams:
    """
    Configuration for the SelCNN model.

    Attributes:
        in_channels (int): Number of input channels for the CNN.
        out_channels (int): Number of output channels for the CNN.
        kernel_size (int): Size of the convolutional kernel.
        padding (int): Padding added to the input.
        dropout_rate (float): Dropout rate for regularization.
        bias_init (float): Initial value for the bias.
        weight_std (float): Standard deviation for weight initialization.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 penalty) for regularization.
        top_k_features (int): Number of top features to select.
        input_size (List[int]): Dimensions of the input (height, width).
    """

    in_channels: int = 512
    out_channels: int = 1
    kernel_size: int = 3
    padding: int = 1
    dropout_rate: float = 0.3
    bias_init: float = 0.0
    weight_std: float = 1e-7
    learning_rate: float = 1e-9
    weight_decay: float = 2.0
    top_k_features: int = 384
    input_size: List[int] = field(default_factory=lambda: [46, 46])


@dataclass
class TrackerParams:
    seq_path: str
    # [x, y, w, h]
    init_bbox: List[float]
    pf_param: ParticleFilterParams
    max_iter_select: int
    max_iter: int
    selcnn_param: SelCNNParams
    roi_size: int = 368
    in_channels: int = 384

    def __post_init__(self):
        self.location = self.init_bbox.copy()
        w, h = self.location[2], self.location[3]
        diag = (w**2 + h**2) ** 0.5
        scale = [diag / w, diag / h]
        self.s1 = [self.pf_param.roi_scale * s for s in scale]
        self.s2 = [self.pf_param.roi_scale * s for s in scale]
        # First image ID is 0
        self.check_num = 19
