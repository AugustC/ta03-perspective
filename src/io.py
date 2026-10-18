import cv2
from src.models import DocumentImage, Point
import xml.etree.ElementTree as ET

CORNER_NAMES = ("tl", "tr", "br", "bl")


def extract_ground_truth(xml_filepath: str) -> dict[str, dict[str, Point]]:
    """
        Extract ground truth from XML file and return a dictionary with frame index as key and point coordinates as values.
        Frames marked as rejected="true" are ignored.

        Parameters:
        xml_filepath (str): Path to the XML file containing ground truth data.

        Returns:
        dict: A dictionary where keys are frame indices (str) and values are dictionaries mapping
              the corner names ('tl', 'tr', 'br', 'bl') to their (x, y) coordinates as floats.
    """
    root = ET.parse(xml_filepath).getroot()
    segmentation_results = root.find('segmentation_results')
    if segmentation_results is None:
        raise ValueError("No segmentation_results found in the XML file.")

    ground_truth = {}
    for frame in segmentation_results.findall('frame'):
        frame_index = frame.get('index')
        if frame.get('rejected', 'false').lower() == 'true':
            continue

        points = {}
        for point in frame.findall('point'):
            point_name = point.get('name')
            x, y = point.get('x'), point.get('y')
            if x is None or y is None:
                raise ValueError(f"Point {point_name} in frame {frame_index} is missing 'x' or 'y' attributes.")
            try:
                points[point_name] = (float(x), float(y))
            except ValueError:
                raise ValueError(f"Invalid coordinates for point {point_name} in frame {frame_index}: ({x}, {y}).") from None

        missing = [name for name in CORNER_NAMES if name not in points]
        if missing:
            raise ValueError(f"Frame {frame_index} is missing the corners {missing}.")

        ground_truth[frame_index] = points
    return ground_truth


def extract_documents(video: cv2.VideoCapture, ground_truth: dict, indices=None) -> list[DocumentImage]:
    """
        Read the video sequentially and return DocumentImage objects for the requested annotated frames.
        Frames are numbered from 1, as in the ground truth. Only the requested frames are kept in memory,
        and frames without annotation are never returned.

        Parameters:
        video (cv2.VideoCapture): OpenCV VideoCapture object for the video file.
        ground_truth (dict): Dictionary containing ground truth data for each frame.
        indices (iterable of str, optional): Frame indices to extract. Defaults to all annotated frames.

        Returns:
        list[DocumentImage]: A list of DocumentImage objects containing frame data and point coordinates,
                             in the order they appear in the video.
    """
    wanted = set(ground_truth) if indices is None else set(indices) & set(ground_truth)

    documents = []
    frame_index = 0
    while len(documents) < len(wanted):
        if not video.grab():  # end of the video
            break
        frame_index += 1
        key = str(frame_index)
        if key not in wanted:
            continue

        ret, frame = video.retrieve()
        if not ret:
            break
        corners = ground_truth[key]
        documents.append(DocumentImage(index=key,
                                       frame=frame,
                                       tl=corners['tl'],
                                       tr=corners['tr'],
                                       bl=corners['bl'],
                                       br=corners['br']))
    return documents
