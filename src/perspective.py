import os
import cv2
import numpy as np
from src.models import DocumentImage
from src.io import extract_ground_truth, extract_documents
import random


def adjust_document(doc: DocumentImage):
    """
        Receives a DocumentImage object, applies a perspective transformation to adjust the document's perspective, and returns the adjusted image.

        Parameters:
        doc (DocumentImage): document containing the original image and the coordinates of the document's corners.

        Returns:
        np.ndarray: The adjusted image with corrected perspective.
    """
    img = doc.frame
    H, W = img.shape[:2]

    points_src = np.array([doc.tl, doc.tr, doc.br, doc.bl], dtype=np.float32)
    points_dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(points_src, points_dst)
    dst = cv2.warpPerspective(img, M, (W, H))

    return dst


def adjust_perspective(video_filepath: str, ground_truth_filepath: str, output_folder: str, n_sample: int = 5):
    """
        Receives a video file path, a ground truth XML file path, an output folder path, and the number of samples to process. It extracts documents from the video based on the ground truth data, applies perspective adjustment to a random sample of documents, and saves the original and adjusted images in the specified output folder.

        Parameters:
        video_filepath (str): Path to the video file.
        ground_truth_filepath (str): Path to the ground truth XML file.
        output_folder (str): Path to the folder where the output images will be saved.
        n_sample (int): Number of random samples to process. Default is 5.

    """
    # Check if the video file exists
    if not os.path.exists(video_filepath):
        err = "'%s' does not exist." % video_filepath
        raise IOError(err)

    # Read data
    video = cv2.VideoCapture(video_filepath)
    ground_truth = extract_ground_truth(ground_truth_filepath)
    documents = extract_documents(video, ground_truth)
    sample = random.sample(documents, min(n_sample, len(documents)))

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create output folder for video file if it doesn't exist
    video_output_folder = os.path.join(output_folder, os.path.basename(video_filepath))
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    # Adjust perspective for each sampled document and save the new image
    for doc in sample:
        new_image = adjust_document(doc)
        cv2.imwrite(os.path.join(video_output_folder, f"{doc.index}_original.png"), doc.frame)
        cv2.imwrite(os.path.join(video_output_folder, f"{doc.index}_adjusted.png"), new_image)

