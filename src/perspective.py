import math
import os
import random
import warnings

import cv2
import numpy as np

from src.models import DocumentImage
from src.io import extract_ground_truth, extract_documents

# Height/width ratio of ISO 216 A-series sheets (A4: 297 mm x 210 mm).
A4_RATIO = math.sqrt(2)


def output_size(frame_shape: tuple, ratio="a4", height: int = 1080) -> tuple[int, int]:
    """
        Compute the (width, height) of the rectified image.

        Parameters:
        frame_shape (tuple): Shape of the original frame (height, width, ...).
        ratio (str or float): "a4" for the A4 proportion (height/width = sqrt(2)), "frame" to keep the
                              size of the original frame (previous behaviour, which distorts the page),
                              or a positive number with the desired height/width ratio.
        height (int): Height of the output image in pixels (ignored when ratio is "frame").

        Returns:
        tuple[int, int]: Width and height of the output image.
    """
    if ratio == "frame":
        return frame_shape[1], frame_shape[0]
    r = A4_RATIO if ratio == "a4" else float(ratio)
    if r <= 0 or height <= 0:
        raise ValueError("ratio and height must be positive.")
    return max(1, round(height / r)), height


def is_valid_quadrilateral(points: np.ndarray) -> bool:
    """
        Check that the corners (tl, tr, br, bl) form a non-degenerate convex quadrilateral in clockwise
        order on the image (y axis pointing down). Crossed or mirrored orders, repeated points and
        missing annotations, such as four (0, 0) corners, fail this test.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if signed_area <= 1.0:
        return False
    return bool(cv2.isContourConvex(pts.astype(np.float32).reshape(-1, 1, 2)))


def adjust_document(doc: DocumentImage, width: int = None, height: int = None):
    """
        Receives a DocumentImage object, applies a perspective transformation to adjust the document's perspective, and returns the adjusted image.

        Parameters:
        doc (DocumentImage): document containing the original image and the coordinates of the document's corners.
        width, height (int, optional): size of the output image. Defaults to the size of the original frame.

        Returns:
        np.ndarray: The adjusted image with corrected perspective.
    """
    img = doc.frame
    if width is None or height is None:
        height, width = img.shape[:2]

    points_src = doc.corners()
    if not is_valid_quadrilateral(points_src):
        raise ValueError(f"Frame {doc.index}: the corners {points_src.tolist()} do not form a valid quadrilateral "
                         "in the order tl, tr, br, bl.")
    points_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(points_src, points_dst)
    dst = cv2.warpPerspective(img, M, (width, height))

    return dst


def sample_frame_indices(ground_truth: dict, n_sample: int, seed=None) -> list[str]:
    """
        Randomly choose up to n_sample annotated frames. The same seed always gives the same frames.
    """
    annotated = sorted(ground_truth, key=int)
    return random.Random(seed).sample(annotated, min(n_sample, len(annotated)))


def adjust_perspective(video_filepath: str, ground_truth_filepath: str, output_folder: str, n_sample: int = 5,
                       seed=None, ratio="a4", height: int = 1080):
    """
        Receives a video file path, a ground truth XML file path, an output folder path, and the number of samples to process. It samples annotated frames of the video, applies perspective adjustment to them, and saves the original and adjusted images in the specified output folder.

        Parameters:
        video_filepath (str): Path to the video file.
        ground_truth_filepath (str): Path to the ground truth XML file.
        output_folder (str): Path to the folder where the output images will be saved.
        n_sample (int): Number of random samples to process. Default is 5.
        seed (int, optional): Seed for the random sampling, for reproducible runs.
        ratio (str or float): Proportion of the output image (see output_size). Default is "a4".
        height (int): Height of the output image in pixels. Default is 1080.

        Returns:
        list[str]: Indices of the frames that were saved.
    """
    # Check if the video file exists
    if not os.path.exists(video_filepath):
        err = "'%s' does not exist." % video_filepath
        raise IOError(err)

    # Choose the frames before reading the video, so that only them are kept in memory
    ground_truth = extract_ground_truth(ground_truth_filepath)
    if not ground_truth:
        raise ValueError("No annotated frames found in the ground truth file.")
    sample = sample_frame_indices(ground_truth, n_sample, seed)

    video = cv2.VideoCapture(video_filepath)
    if not video.isOpened():
        raise IOError("Could not open '%s'." % video_filepath)
    try:
        documents = extract_documents(video, ground_truth, sample)
    finally:
        video.release()

    missing = sorted(set(sample) - {doc.index for doc in documents}, key=int)
    if missing:
        warnings.warn(f"Frames {missing} are annotated but could not be read from the video.")

    # Create output folder for video file if it doesn't exist
    video_output_folder = os.path.join(output_folder, os.path.basename(video_filepath))
    os.makedirs(video_output_folder, exist_ok=True)

    # Adjust perspective for each sampled document and save the new image
    saved = []
    for doc in documents:
        width, out_height = output_size(doc.frame.shape, ratio, height)
        try:
            new_image = adjust_document(doc, width, out_height)
        except ValueError as e:
            warnings.warn(str(e))
            continue
        cv2.imwrite(os.path.join(video_output_folder, f"{doc.index}_original.png"), doc.frame)
        cv2.imwrite(os.path.join(video_output_folder, f"{doc.index}_adjusted.png"), new_image)
        saved.append(doc.index)
    return saved
