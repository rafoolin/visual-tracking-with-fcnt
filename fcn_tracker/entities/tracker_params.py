from dataclasses import dataclass
from typing import List

from entities.particle_filter_params import ParticleFilterParams


@dataclass
class TrackerParams:
    seq_path: str
    # [x, y, w, h]
    init_bbox: List[float]
    pf_param: ParticleFilterParams
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
