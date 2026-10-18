from dataclasses import dataclass
import numpy as np

Point = tuple[float, float]


@dataclass
class DocumentImage:
    index: str
    frame: np.ndarray
    tl: Point
    tr: Point
    bl: Point
    br: Point

    def corners(self) -> np.ndarray:
        """
            Returns the document corners as a 4x2 float32 array, in the order expected by
            cv2.getPerspectiveTransform in this project: top-left, top-right, bottom-right, bottom-left.
        """
        return np.array([self.tl, self.tr, self.br, self.bl], dtype=np.float32)
