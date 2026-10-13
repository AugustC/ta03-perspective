from dataclasses import dataclass
import numpy as np


@dataclass
class DocumentImage:
    index: str
    frame: np.ndarray
    tl: tuple[int, int]
    tr: tuple[int, int]
    bl: tuple[int, int]
    br: tuple[int, int]

