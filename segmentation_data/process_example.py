import os

def _set_path():
    # Add the root of the venv to the path
    # This is necessary to import the selective_search module

    # Get the path to the venv
    venv_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Add the venv path to the path
    import sys
    sys.path.append(venv_path)

_set_path()

from preprocess_data.generate_selective_search_examples import _convert_ss_boxes_to_coco, _convert_coco_bounds_to_PIL_bounds
from selective_search import selective_search_fast
import PIL
import numpy as np


def process_example(example, classes):
    '''
    Process a single example.

    Args:
        example: One coco datapoint from fiftyone.
        classes: A list of classes to process.

    Returns:
        A list of dictionaries. Each dictionary has the following keys:
        - image: A numpy array of shape (height, width, 3) and dtype uint8.
        - mask: A numpy array of shape (height, width) and dtype uint8.
        - label: A string representing the class label.
    
    Notes:
        The image is a numpy array of shape (height, width, 3) and dtype
        uint8. The mask is a numpy array of shape (height, width) and dtype
        uint8. The mask has values 0 and 1, where 0 is background and 1 is
        foreground.

        The image is cropped using selective search. A box is chosen with
        a few criteria:
        - The box encompasses the entire object
        - The box is not too large

        The mask is expanded to fill the box. Any pixels that are not
        from the original mask are set to 0.
    '''

    # Load the image as a numpy array
    image = _load_image(example)

    # Get the selective search boxes
    ss_boxes = _get_boxes(example)

    # Coco stores the bounds as [y0, x0, y1, x1] and as a ratio of the image size.
    # The selective search boxes are [x, y, w, h]. It is stored in pixel values.
    # Convert the selective search boxes to the same format as Coco.
    ss_boxes = _convert_ss_boxes_to_coco(ss_boxes, image.size)

    # For each object, pick the box that best fits the object,
    # crop the image, and expand the mask to fit the box.
    # Only process objects in the classes list.
    objects = []
    for detection in example["ground_truth"]["detections"]:
        label = detection["label"]
        if label not in classes:
            continue

        mask = detection["mask"]
        bbox = detection["bounding_box"]
        
        # Get the box that best fits the object
        box = _pick_ss_box(ss_boxes, bbox)

        # Crop the image
        cropped_image = _crop_image(image, box)

        # Expand the mask
        expanded_mask = _expand_mask(mask, bbox, box, image.size)

        # Add the object to the list
        objects.append({
            "image": cropped_image,
            "mask": expanded_mask,
            "label": label
        })

    return objects

def _load_image(example):
    '''
    Load the image from the example.

    Args:
        example: One coco datapoint from fiftyone.

    Returns:
        A PIL image.
    '''

    # Get the image path
    image_path = example["filepath"]

    # Load the image
    image = PIL.Image.open(image_path)

    return image

def _get_boxes(example):
    '''
    Get the selective search boxes for the example.

    Args:
        example: One coco datapoint from fiftyone.

    Returns:
        A list of boxes. Each box is a list of four numbers in the selective search format:
        (x, y, w, h), where the numbers are in pixel values.
    '''

    # Get the image path
    image_path = example["filepath"]

    # Run selective search
    boxes = selective_search_fast(image_path)

    return boxes

def _pick_ss_box(ss_boxes, bbox):
    '''
    Picks a selective search box that encompasses the object.

    Args:
        ss_boxes: A list of boxes from selective search.
        bbox: A list of four numbers representing the bounding box of the object.

    Returns:
        A list of four numbers representing the box that best fits the object.
    
    Notes:
        The box is chosen with a few criteria:
        - The box encompasses the entire object
        - The box is not too large

        The input boxes should be in the Coco format ([y0, x0, y1, x1], ratio of image size).
        The output box is the same.
    '''

    

    # Iterate over the ss_boxes. For each box, check if it encompasses the object.
    # If it does, and is not too large, return it.
    index = 0
    for box in ss_boxes:
        # Check if the box encompasses the object
        if _box_encompasses_other_box(box, bbox):
            # Check if the box is too large
            if _box_is_not_too_large(box, bbox):
                print("Picked box and index is:", index)
                return box
            
        index += 1
            
    # If no box is found, return the original box
    return bbox

def _box_encompasses_other_box(a, b):
    '''
    Check if box a encompasses box b.

    Args:
        a: A list of four numbers representing a box.
        b: A list of four numbers representing a box.
    
    Returns:
        True if box a encompasses box b. False otherwise.

    Notes:
        The boxes are in the format [y0, x0, y1, x1].
    '''

    # Check if the x0 and y0 of b are inside a
    if a[0] <= b[0] and a[1] <= b[1]:
        # Check if the x1 and y1 of b are inside a
        if a[2] >= b[2] and a[3] >= b[3]:
            return True
    
    return False

def _box_is_not_too_large(a, b):
    '''
    Check if box a is not too large compared to box b.

    Args:
        a: A list of four numbers representing a box.
        b: A list of four numbers representing a box.
    
    Returns:
        True if box a is not too large compared to box b. False otherwise.

    Notes:
        The boxes are in the format [y0, x0, y1, x1].
        "Not too large" means that the area of box a is less than twice the area of box b.
    '''
    # Get the area of box a
    area_a = (a[2] - a[0]) * (a[3] - a[1])

    # Get the area of box b
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    # Get the ratio of the areas
    ratio = area_a / area_b

    # If the ratio is less than 2, return True
    if ratio < 2:
        return True
    
    return False

def _crop_image(image, box):
    '''
    Crop the image using the box.

    Args:
        image: A PIL image.
        box: A list of four numbers representing a box. The box is in the Coco format ([y0, x0, y1, x1], ratio of image size).
    
    Returns:
        A numpy array of shape (height, width, 3) and dtype uint8.
    '''
    pil_box = _convert_coco_bounds_to_PIL_bounds(box, image.size)

    # Crop the image
    cropped_image = image.crop(pil_box)

    # Convert the image to a numpy array
    cropped_image = np.array(cropped_image)

    return cropped_image

def _expand_mask(mask, mask_border, dest_box, image_shape):
    '''
    Expand the mask to fit the box.

    Args:
        mask: A numpy array of shape (height, width) and dtype uint8.
        mask_location: A list of four numbers representing the bounding box of the mask. The box is in the Coco format ([y0, x0, y1, x1], ratio of image size).
        box: A list of four numbers representing a box. The box is in the Coco format ([y0, x0, y1, x1], ratio of image size).

    Returns:
        A numpy array of shape (height, width) and dtype uint8.
    
    Notes:
        The mask is expanded by a few pixels to fit the box.
    '''

    # Convert the mask location to a PIL box
    mask_border = _convert_coco_bounds_to_PIL_bounds(mask_border, image_shape)

    # Convert the box to a PIL box
    dest_box = _convert_coco_bounds_to_PIL_bounds(dest_box, image_shape)

    # Convert dest_box to integers
    dest_box = [int(x) for x in dest_box]

    # Create an empty mask of the size of the box
    expanded_mask = np.zeros((dest_box[3] - dest_box[1], dest_box[2] - dest_box[0]), dtype=np.uint8)

    # Identify where the mask is in the box
    # The format is (x0, y0, x1, y1)
    mask_location_in_box = mask_border[0] - dest_box[0], mask_border[1] - dest_box[1], mask_border[2] - dest_box[0], mask_border[3] - dest_box[1]

    # Add the mask to the expanded mask
    for y_pos in range(mask_location_in_box[1], mask_location_in_box[3]):
        for x_pos in range(mask_location_in_box[0], mask_location_in_box[2]):
            expanded_mask[y_pos, x_pos] = mask[y_pos - mask_location_in_box[1], x_pos - mask_location_in_box[0]]


    return expanded_mask