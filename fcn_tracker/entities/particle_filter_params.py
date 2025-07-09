from dataclasses import dataclass
from typing import List
import math


@dataclass
class ParticleFilterParams:
    # [dx, dy, scale, aspect, rotation, skew]
    affsig: List[float]
    p_sz: int
    p_num: int
    mv_thr: float
    up_thr: float
    roi_scale: float
    init_bbox: List[float]

    def __post_init__(self):
        location = self.init_bbox.copy()
        # Adjust affsig based on object size
        w, h = location[2], location[3]
        diag = (w**2 + h**2) ** 0.5

        self.affsig[0] = self.affsig[1] = math.ceil(diag / 7)
        self.ratio = w / self.p_sz

        self.affsig[2] *= self.ratio
        self.affsig_o = self.affsig.copy()

        self.affsig[2] = 0
        self.minconf = 0.5
