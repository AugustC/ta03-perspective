import cv2
from src.models import DocumentImage
import xml.etree.ElementTree as ET

def extract_ground_truth(xml_filepath: str) -> dict:
    """
        Extract ground truth from XML file and return a dictionary with frame index as key and point coordinates as values.

        Parameters:
        xml_filepath (str): Path to the XML file containing ground truth data.

        Returns:
        dict: A dictionary where keys are frame indices and values are dictionaries of point names and their (x, y) coordinates.
    """
    root = ET.parse(xml_filepath).getroot()
    segmentation_results = root.find('segmentation_results')
    if segmentation_results is None:
        raise ValueError("No segmentation_results found in the XML file.")

    ground_truth = {}
    for frame in segmentation_results.findall('frame'):
        ground_truth[frame.get('index')] = {}
        for point in frame.findall('point'):
            point_name = point.get('name')
            try:
                x = point.get('x')
                y = point.get('y')
            except AttributeError:
                raise AttributeError(f"Point {point_name} in frame {frame.get('index')} is missing 'x' or 'y' attributes.")
            except ValueError:
                raise ValueError(f"Invalid coordinates for point {point_name} in frame {frame.get('index')}.")

            ground_truth[frame.get('index')][point_name] = [x, y]
    return ground_truth


def extract_documents(video: cv2.VideoCapture, ground_truth: dict) -> list[DocumentImage]:
    """
        Extract documents from video frames and return a list of DocumentImage objects.

        Parameters:
        video (cv2.VideoCapture): OpenCV VideoCapture object for the video file.
        ground_truth (dict): Dictionary containing ground truth data for each frame.

        Returns:
        list[DocumentImage]: A list of DocumentImage objects containing frame data and point coordinates.
    """

    documents = []
    ret = True
    frame_index = 0
    while ret:
        ret, frame = video.read()
        frame_index += 1
        doc = DocumentImage(index = str(frame_index),
                            frame= frame,
                            tl = ground_truth.get(str(frame_index), {}).get('tl', (0, 0)),
                            tr = ground_truth.get(str(frame_index), {}).get('tr', (0, 0)),
                            bl = ground_truth.get(str(frame_index), {}).get('bl', (0, 0)),
                            br = ground_truth.get(str(frame_index), {}).get('br', (0, 0)))
        documents.append(doc)

    return documents
