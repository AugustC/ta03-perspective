import os
import tempfile
import unittest

import cv2
import numpy as np

from src.io import extract_ground_truth, extract_documents
from src.models import DocumentImage
from src.perspective import (A4_RATIO, adjust_document, adjust_perspective, is_valid_quadrilateral,
                             output_size, sample_frame_indices)

# Corners of a page as seen in a 1920x1080 frame (tl, tr, br, bl), similar to SmartDoc's datasheet001.
QUAD = np.float32([[737.3, 198.5], [1181.1, 196.9], [1257.2, 788.8], [715.1, 801.5]])


def write_xml(path, frames):
    """frames: {index: corners (4x2, order tl, tr, br, bl) or None for a rejected frame}"""
    lines = ['<?xml version="1.0" encoding="utf-8"?>', '<seg_result version="0.2">', '<segmentation_results>']
    for index, corners in frames.items():
        if corners is None:
            lines.append(f'<frame index="{index}" rejected="true"/>')
            continue
        lines.append(f'<frame index="{index}" rejected="false">')
        for name, (x, y) in zip(["tl", "tr", "br", "bl"], corners):
            lines.append(f'<point name="{name}" x="{x:.4f}" y="{y:.4f}"/>')
        lines.append('</frame>')
    lines += ['</segmentation_results>', '</seg_result>']
    with open(path, "w") as f:
        f.write("\n".join(lines))


def synthetic_page(width=420, height=594):
    """A4 page (2 px per mm) with large blocks, so that two interpolations change it little."""
    page = np.full((height, width, 3), 235, np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(25):
        x, y = rng.integers(20, width - 80), rng.integers(20, height - 60)
        cv2.rectangle(page, (int(x), int(y)), (int(x) + 60, int(y) + 40), tuple(int(c) for c in rng.integers(0, 200, 3)), -1)
    return page


def render_frame(page, quad, size=(1920, 1080)):
    h, w = page.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]), quad)
    frame = np.full((size[1], size[0], 3), (150, 120, 110), np.uint8)
    warped = cv2.warpPerspective(page, M, size)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, size) > 127
    frame[mask] = warped[mask]
    return frame


class TestGroundTruth(unittest.TestCase):
    def test_parses_floats_and_skips_rejected_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gt.xml")
            write_xml(path, {1: QUAD, 2: None, 3: QUAD + 1})
            gt = extract_ground_truth(path)
        self.assertEqual(sorted(gt), ["1", "3"])
        self.assertIsInstance(gt["1"]["tl"][0], float)
        self.assertAlmostEqual(gt["3"]["br"][1], 789.8, places=3)

    def test_missing_coordinate_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gt.xml")
            with open(path, "w") as f:
                f.write('<r><segmentation_results><frame index="1"><point name="tl" x="1"/></frame>'
                        '</segmentation_results></r>')
            with self.assertRaises(ValueError):
                extract_ground_truth(path)


class TestGeometry(unittest.TestCase):
    def test_output_size(self):
        self.assertEqual(output_size((1080, 1920, 3), "a4", 1080), (764, 1080))
        self.assertEqual(output_size((1080, 1920, 3), "frame", 500), (1920, 1080))
        self.assertEqual(output_size((1080, 1920, 3), 1.0, 300), (300, 300))
        self.assertAlmostEqual(1080 / 764, A4_RATIO, places=2)

    def test_quadrilateral_validation(self):
        tl, tr, br, bl = QUAD
        self.assertTrue(is_valid_quadrilateral(QUAD))
        self.assertFalse(is_valid_quadrilateral(np.zeros((4, 2))))              # frame without annotation
        self.assertFalse(is_valid_quadrilateral(np.array([tl, br, tr, bl])))    # crossed
        self.assertFalse(is_valid_quadrilateral(np.array([tr, tl, bl, br])))    # mirrored

    def test_rectification_recovers_the_page(self):
        page = synthetic_page()
        frame = render_frame(page, QUAD)
        doc = DocumentImage("1", frame, *(tuple(map(float, p)) for p in QUAD[[0, 1, 3, 2]]))
        width, height = output_size(frame.shape, "a4", page.shape[0])
        self.assertEqual((width, height), (page.shape[1], page.shape[0]))
        out = adjust_document(doc, width, height)
        diff = np.abs(out.astype(int) - page.astype(int))[10:-10, 10:-10]
        self.assertLess(diff.mean(), 6.0)

    def test_invalid_corners_raise(self):
        doc = DocumentImage("7", np.zeros((1080, 1920, 3), np.uint8), (0, 0), (0, 0), (0, 0), (0, 0))
        with self.assertRaises(ValueError):
            adjust_document(doc)

    def test_sampling_is_reproducible_and_only_uses_annotated_frames(self):
        gt = {str(i): {} for i in (2, 5, 9, 14, 20, 33)}
        a = sample_frame_indices(gt, 3, seed=42)
        self.assertEqual(a, sample_frame_indices(gt, 3, seed=42))
        self.assertTrue(set(a) <= set(gt))
        self.assertEqual(len(sample_frame_indices(gt, 50, seed=1)), len(gt))


class TestVideo(unittest.TestCase):
    N_FRAMES = 12

    def make_video(self, tmp):
        path = os.path.join(tmp, "doc.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (1920, 1080))
        self.assertTrue(writer.isOpened(), "MJPG writer unavailable")
        page = synthetic_page()
        for i in range(1, self.N_FRAMES + 1):
            frame = render_frame(page, QUAD)
            cv2.putText(frame, str(i), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
            frame[1000:, :200] = 20 * i          # frame number encoded as a gray level
            writer.write(frame)
        writer.release()
        return path

    def test_extract_documents_returns_only_requested_annotated_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = self.make_video(tmp)
            gt = {str(i): {"tl": (0, 0), "tr": (1, 0), "br": (1, 1), "bl": (0, 1)} for i in (2, 5, 9, 12)}
            video = cv2.VideoCapture(video_path)
            docs = extract_documents(video, gt, ["5", "12", "13"])   # 13 is not annotated
            video.release()
        self.assertEqual([d.index for d in docs], ["5", "12"])
        for d in docs:
            self.assertIsNotNone(d.frame)
            self.assertAlmostEqual(d.frame[1020:, :180].mean(), 20 * int(d.index), delta=6)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = self.make_video(tmp)
            xml_path = os.path.join(tmp, "doc.xml")
            write_xml(xml_path, {i: QUAD for i in range(1, self.N_FRAMES + 1)})
            out = os.path.join(tmp, "outs")
            saved = adjust_perspective(video_path, xml_path, out, n_sample=4, seed=3)
            self.assertEqual(saved, sorted(saved, key=int))
            self.assertEqual(len(saved), 4)
            self.assertEqual(saved, adjust_perspective(video_path, xml_path, out, n_sample=4, seed=3))
            adjusted = cv2.imread(os.path.join(out, "doc.avi", f"{saved[0]}_adjusted.png"))
            self.assertEqual(adjusted.shape, (1080, 764, 3))


if __name__ == "__main__":
    unittest.main()
