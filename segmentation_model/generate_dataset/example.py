from PIL import Image
from conversions import convert_box_type
import numpy as np

class Example():
    '''
    A class that represents an example from the dataset.

    Attributes:
        example: The example from the dataset.
        detections: The detections in the example.

    Methods:
        `get_mask_for(detection_index: int)` -> `numpy.ndarray`:
            Get the mask for the detection.
        `get_image()` -> `PIL.Image`:
            Get the image.
        `get_label_for(detection_index: int)` -> `str`:
            Get the label for the detection.
        `get_bounding_box_for(detection_index: int)` -> `numpy.ndarray`:
            Get the bounding box for the detection.
        `get_bounding_boxes()` -> `list`:
            Get the bounding boxes for all the detections.
        `get_labels()` -> `list`:
            Get the labels for all the detections.
        `get_cropped_image()` -> `PIL.Image`:
            Get the cropped image.
        `get_path_to_image()` -> `str`:
            Get the path to the image.
        `get_image_size()` -> `tuple[int]`:
            Get the image size.
        `get_masks()` -> `list[numpy.ndarray]`:
            Get the masks for all the detections.
    
    Notes:
        The bounding box is a list of four numbers: [x, y, w, h].
        The mask is a numpy array of shape (height, width) and is of a boolean type.
    '''

    def __init__(self, example):
        self.example = example
        self.detections = example["ground_truth"]['detections']
    
    def get_mask_for(self, detection_index: int):
        '''
        Get the mask for the detection.

        Args:
            detection_index: The index of the detection.

        Returns:
            A numpy array of the mask.

        Raises:
            IndexError: If the detection index is out of range.

        Notes:
            The mask is a numpy array of shape (height, width) and is of a boolean type.
        '''

        # If the index is out of range, raise an error
        self._check_index(detection_index)

        # Get the detection
        detection = self.detections[detection_index]

        # Get the mask
        mask = detection['mask']

        return mask
    
    def get_masks(self) -> list[np.ndarray]:
        '''
        Get the masks for all the detections.
        
        Returns:
            A list of numpy arrays of the masks.

        Notes:
            The mask is of a boolean type.
        '''

        # Get the masks
        masks = [detection['mask'] for detection in self.detections]

        return masks

    def get_image(self):
        '''
        Get the image.

        Returns:
            A PIL image.

        Notes:
            The image is a PIL image.
        '''

        # Get the image path and open it with PIL
        image_path = self.example["filepath"]

        image = Image.open(image_path)

        return image
    
    def get_image_size(self) -> tuple[int]:
        '''
        Get the image size.

        Returns:
            A tuple of the image size: (width, height).
        '''

        # Get the image
        image = self.get_image()

        # Get the image size
        image_size = image.size

        return image_size
    

    def get_label_for(self, detection_index: int):
        '''
        Get the label for a specific detection.

        Args:
            detection_index: The index of the detection.
        
        Returns:
            The label for the detection.
        
        Raises:
            IndexError: If the detection index is out of range.
        '''

        # If the index is out of range, raise an error
        self._check_index(detection_index)

        # Get the detection
        detection = self.detections[detection_index]

        # Get the label
        label = detection['label']

        return label


    def get_bounding_box_for(self, detection_index: int):
        '''
        Get the bounding box for a specific detection.

        Args:
            detection_index: The index of the detection.
        
        Returns:
            The bounding box for the detection.

        Raises:
            IndexError: If the detection index is out of range.
        
        Notes:
            The bounding box is a list of four numbers: [x, y, w, h].
            The numbers are a ratio of the image size.
        '''

        # If the index is out of range, raise an error
        self._check_index(detection_index)

        # Get the detection
        detection = self.detections[detection_index]
        
        # Get the bounding box
        bounding_box = detection['bounding_box']

        return bounding_box

    def get_bounding_boxes(self) -> list[list[float]]:
        '''
        Get the bounding boxes for all detections.

        Returns:
            A list of bounding boxes. Each bounding box is a list of four numbers: [x, y, w, h].
            The numbers are a ratio of the image size.
        '''

        # Get the bounding boxes
        bounding_boxes = [detection['bounding_box'] for detection in self.detections]

        return bounding_boxes

    def get_labels(self) -> list[str]:
        '''
        Get the labels for all detections.

        Returns:
            A list of labels.
        '''

        # Get the labels
        labels = [detection["label"] for detection in self.detections]

        return labels

    def get_cropped_image(self, bounding_box: list):
        '''
        Get the image cropped to the bounding box.

        Args:
            bounding_box: A list of four numbers representing the bounding box of the object. The box should be in the Coco format ([x, y, w, h], ratio of image size).

        Returns:
            A PIL image.

        Notes:
            The image is a PIL image.
        '''

        # Get the image
        image = self.get_image()

        # Get the image size
        image_size = image.size

        # Convert the bounding box to the PIL format
        bounding_box = convert_box_type(bounding_box, "coco", "pil", image_size)
        
        # Crop the image
        cropped_image = image.crop(bounding_box)

        return cropped_image

    def get_path_to_image(self) -> str:
        '''
        Get the path to the image.

        Returns:
            The path to the image.
        '''

        # Get the image path
        image_path = self.example["filepath"]

        return image_path
    
    def get_detection_id_(self, detection_index: int) -> str:
        '''
        Get the detection id for the detection.

        Args:
            detection_index: The index of the detection.

        Returns:
            The detection id.

        Raises:
            IndexError: If the detection index is out of range.
        '''

        # If the index is out of range, raise an error
        self._check_index(detection_index)

        # Get the detection
        detection = self.detections[detection_index]

        # Get the detection id
        detection_id = detection['id']

        return detection_id
    
    def get_detection_ids(self) -> list[str]:
        '''
        Get the detection ids for all detections.

        Returns:
            A list of detection ids.
        '''

        # Get the detection ids
        detection_ids = [detection['id'] for detection in self.detections]

        return detection_ids

    def _check_index(self, index):
        '''
        Check if the index is in range.

        Args:
            index: The index to check.
            example: The example to check.

        Raises:
            IndexError: If the index is out of range.
        '''

        # Get the number of detections
        num_detections = len(self.detections)

        # If the index is out of range, raise an error
        if index < 0 or index >= num_detections:
            raise IndexError