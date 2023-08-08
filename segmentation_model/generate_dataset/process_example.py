import os

def _set_path():
    # Add the root of the venv to the path
    # This is necessary to import the selective_search module

    # Get the path to the venv
    venv_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Add the venv path to the path
    import sys
    sys.path.append(venv_path)

_set_path()

from inference.selective_search import selective_search_fast
import PIL
import numpy as np

from segmentation_model.generate_dataset.example import Example
from segmentation_model.generate_dataset.conversions import convert_box_type


def process_example(input_example, classes, use_selective_search: bool = True, min_size = (30, 30)):
    '''
    Process a single example.

    Args:
        example: One coco datapoint from fiftyone.
        classes: A list of classes to process.

    Returns:
        A list of dictionaries. Each dictionary has the following keys:
        - image: A PIL image.
        - mask: A numpy array of shape (height, width) and dtype uint8.
        - label: A string representing the class label.
        - id: The detection id.
    
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
    example = Example(input_example)

    # Load the image as a numpy array
    image = example.get_image()

    ss_boxes = None
    if use_selective_search:
        # Get the selective search boxes
        ss_boxes = selective_search_fast(example.get_path_to_image())

        # Coco stores the bounds as [x, y, w, h] and as a ratio of the image size.
        # The selective search boxes are [x, y, w, h]. It is stored in pixel values.
        # Convert the selective search boxes to the same format as Coco.
        for ss_box in ss_boxes:
            ss_box = convert_box_type(ss_box, "selective_search", "coco", example.get_image_size())

    # For each object, pick the box that best fits the object,
    # crop the image, and expand the mask to fit the box.
    # Only process objects in the classes list.
    objects = []
    for label, bbox, mask, detection_id in zip(example.get_labels(), example.get_bounding_boxes(), example.get_masks(), example.get_detection_ids()):
        # Only process objects in the classes list
        if label not in classes:
            continue
        
        # Get the box that best fits the object
        box = bbox
        if use_selective_search:
            box = _pick_ss_box(ss_boxes, bbox)

        # Crop the image
        cropped_image = example.get_cropped_image(box)

        # Expand the mask
        expanded_mask = None
        if use_selective_search:
            expanded_mask = _expand_mask(mask, bbox, box, example.get_image_size())
        else:
            expanded_mask = mask.copy()

        expanded_mask = expanded_mask.astype(np.uint8)
        expanded_mask *= 255

        # Add the object to the list
        objects.append({
            "image": cropped_image,
            "mask": expanded_mask,
            "label": label,
            "id": detection_id
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

        The input boxes should be in the Coco format ([x, y, w, h], ratio of image size).
        The output box is the same.
    '''

    

    # Iterate over the ss_boxes. For each box, check if it encompasses the object.
    # If it does, and is not too large, return it.
    for box in ss_boxes:
        # Check if the box encompasses the object
        if _box_encompasses_other_box(box, bbox):
            # Check if the box is too large
            if _box_is_not_too_large(box, bbox):
                print("Found a box")
                return box
            
    # If no box is found, return the original box
    return bbox

def _box_encompasses_other_box(input_a, input_b):
    '''
    Check if box a encompasses box b.

    Args:
        a: A list of four numbers representing a box.
        b: A list of four numbers representing a box.
    
    Returns:
        True if box a encompasses box b. False otherwise.

    Notes:
        The boxes are in the format [x, y, w, h].
    '''

    # Convert the boxes to the format [x0, y0, x1, y1]
    a = [input_a[0], input_a[1], input_a[0] + input_a[2], input_a[1] + input_a[3]]
    b = [input_b[0], input_b[1], input_b[0] + input_b[2], input_b[1] + input_b[3]]

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
        The boxes are in the format [x, y, w, h].
        "Not too large" means that the area of box a is less than twice the area of box b.
    '''
    EXPANSION_CONSTANT = 1

    # Get the area of box a
    area_a = a[2] * a[3]

    # Get the area of box b
    area_b = b[2] * b[3]

    # Get the ratio of the areas
    ratio = area_a / area_b

    # If the ratio is less than 2, return True
    if ratio < EXPANSION_CONSTANT:
        return True
    
    return False

def _expand_mask(mask, mask_border, dest_box, image_shape):
    '''
    Expand the mask to fit the box.

    Args:
        mask: A numpy array of shape (height, width) and dtype uint8.
        mask_location: A list of four numbers representing the bounding box of the mask. The box is in the Coco format ([x, y, w, h], ratio of image size).
        box: A list of four numbers representing a box. The box is in the Coco format ([x, y, w, h], ratio of image size).
        image_shape: A list of two numbers representing the shape of the image in the format (height, width).

    Returns:
        A numpy array of shape (height, width) and dtype uint8.
    
    Notes:
        The mask is expanded by a few pixels to fit the box.
    '''
    # Convert the mask location to a PIL box
    mask_border = convert_box_type(mask_border, "coco", "pil", image_shape)

    # Convert the box to a PIL box
    dest_box = convert_box_type(dest_box, "coco", "pil", image_shape)

    # Create an empty mask of the size of the box
    expanded_mask = np.zeros((dest_box[3] - dest_box[1], dest_box[2] - dest_box[0]), dtype=np.uint8)

    # Make sure the expanded mask is at least the size of the mask
    if expanded_mask.shape[0] < mask.shape[0]:
        expanded_mask = np.zeros((mask.shape[0], expanded_mask.shape[1]), dtype=np.uint8)
    if expanded_mask.shape[1] < mask.shape[1]:
        expanded_mask = np.zeros((expanded_mask.shape[0], mask.shape[1]), dtype=np.uint8)

    # Apply the mask to the expanded mask
    # The first step is to identify where in the expanded mask the original mask is to lie.
    # This will be tracked with an offset coordinate.
    # For example:
    #   mask_border =   [110, 55, 150, 100]
    #   dest_box =      [100, 50, 200, 200]
    #   The offset is (10, 5)
    #
    # To simplify the process of copying the mask onto the expanded mask with the offset, we will use
    # a function: _assign_with_offset(matrix = expanded_mask, offset = (10, 5), index = (i, j), value = mask[i][j]).

    def _assign_with_offset(matrix, offset, input_index, value):
        '''
        Assign a value to a matrix with an offset.

        Args:
            matrix: A numpy array.
            offset: A tuple of two numbers representing the offset.
            index: A tuple of two numbers representing the index of the matrix to assign the value to.
            value: The value to assign to the matrix.

        Returns:
            None
        
        Notes:
            The matrix is modified in place.
            The format of the offset and index is (y, x).
        '''
        index = list(input_index)
        index[0] += offset[0]
        index[1] += offset[1]

        matrix[index[0]][index[1]] = value

    # Calculate the offset
    offset = (mask_border[1] - dest_box[1], mask_border[0] - dest_box[0])

    # Iterate over the mask. The format of the mask shape is (height, width).
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # Assign the value of the mask to the expanded mask
            try:
                _assign_with_offset(expanded_mask, offset, (i, j), mask[i][j])
            except IndexError:
                # Print a diagnostic message with the sizes of the mask and the expanded mask and the offset
                print("Error: The mask is too large for the expanded mask.")
                print("Mask shape: {}".format(mask.shape))
                print("Expanded mask shape: {}".format(expanded_mask.shape))
                print("Offset: {}".format(offset))


    return expanded_mask