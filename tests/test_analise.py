import unittest

import cv2
import numpy as np

from analise.geometry import measure, zhang_he


def project_a4(rotation, focal=1500.0, size=(1920, 1080), distance=600.0):
    """Projeta uma folha A4 (210 x 297 mm) com uma câmera pinhole; devolve tl, tr, br, bl na imagem."""
    w, h = 210.0, 297.0
    K = np.array([[focal, 0, (size[0] - 1) / 2], [0, focal, (size[1] - 1) / 2], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.asarray(rotation, float))
    P = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    x = (K @ (R @ P.T + np.array([[0], [0], [distance]]))).T
    return x[:, :2] / x[:, 2:]


class TestZhangHe(unittest.TestCase):
    def test_recovers_focal_length_and_a4_ratio(self):
        for rotation in ([0.12, 0.29, 0.12], [0.16, -0.19, 0.2], [0.3, 0.05, -0.1]):
            tl, tr, br, bl = project_a4(rotation)
            f, ratio = zhang_he(tl, tr, br, bl, (1920 - 1) / 2, (1080 - 1) / 2)
            self.assertAlmostEqual(f, 1500.0, delta=1.0)
            self.assertAlmostEqual(ratio, 297 / 210, places=4)

    def test_side_lengths_underestimate_a_tilted_page(self):
        r = measure(project_a4([0.35, 0.08, 0.0]), 1920, 1080)
        self.assertLess(r["hw_mean_sides"], 297 / 210 * 0.95)
        self.assertAlmostEqual(r["hw_zhang_he"], 297 / 210, places=3)

    def test_parallel_sides_cannot_determine_focal_length(self):
        # inclinação só para a frente: os lados superior e inferior continuam paralelos na imagem
        tl, tr, br, bl = project_a4([0.35, 0.0, 0.0])
        self.assertEqual(zhang_he(tl, tr, br, bl, (1920 - 1) / 2, (1080 - 1) / 2), (None, None))


if __name__ == "__main__":
    unittest.main()
