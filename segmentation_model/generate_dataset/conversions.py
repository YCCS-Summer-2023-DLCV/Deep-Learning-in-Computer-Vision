'''
This file contains functions for converting bounding boxes between different formats.

The valid types are: "coco", "pil", "selective_search". It is not case sensitive.

Coco format is [x, y, w, h], ratio of image size.
PIL format is [x0, y0, x1, y1], pixels.
Selective search format is [x, y, w, h], pixels.

Functions:
    convert_box_type: Convert a bounding box from one type to another. Raises an error if the image_size is required but not provided.
'''

def convert_box_type(input_box, box_type: str, to_type: str, image_size = None):
    '''
    Convert a bounding box from one type to another. Raises an error if the image_size is required but not provided.

    Args:
        box: The bounding box to convert.
        box_type: The type of the bounding box.
        to_type: The type to convert to.
        image_size: The size of the image. Only required if converting between one type and Coco format.

    Returns:
        The bounding box in the new format.

    Raises:
        ValueError: If the box or to type is not recognized.
        ValueError: If the image size is not provided when converting to or from Coco format.

    Notes:
        Coco format is [x, y, w, h], ratio of image size.
        PIL format is [x0, y0, x1, y1], pixels.
        Selective search format is [x, y, w, h], pixels.

        The image size is a tuple of (width, height, ...).

        The valid types are: "coco", "pil", "selective_search". It is not case sensitive.
    '''

    # Copy the box
    box = input_box.copy()

    # Convert the types to lowercase
    box_type = box_type.lower()
    to_type = to_type.lower()

    # Check if the box and to type is valid
    if box_type not in ["coco", "pil", "selective_search"]:
        raise ValueError("The box type is not recognized.")
    if to_type not in ["coco", "pil", "selective_search"]:
        raise ValueError("The to type is not recognized.")

    # If they are the same, return the box
    if box_type == to_type:
        return box

    # First convert to PIL format
    if box_type == "coco":
        # If the image size is not provided, raise an error
        if image_size is None:
            raise ValueError("The image size must be provided when converting to or from Coco format.")

        # Convert the box to PIL format
        box = _convert_coco_to_pil(box, image_size)

    elif box_type == "selective_search":
        # Convert the box to PIL format
        box = _convert_selective_search_to_pil(box)

    # Then convert to the new format
    if to_type == "coco":
        # If the image size is not provided, raise an error
        if image_size is None:
            raise ValueError("The image size must be provided when converting to or from Coco format.")

        # Convert the box to Coco format
        box = _convert_pil_to_coco(box, image_size)

    elif to_type == "selective_search":
        # Convert the box to selective search format
        box = _convert_pil_to_selective_search(box)

    return box

def _convert_coco_to_pil(input_box, image_size):
    '''
    Convert a bounding box from Coco format to PIL format.

    Args:
        box: The bounding box to convert.
        image_size: The size of the image.

    Returns:
        The bounding box in PIL format.

    Notes:
        Coco format is [x, y, w, h], ratio of image size.
        PIL format is [x0, y0, x1, y1], pixels.
    '''

    # Get the image size
    image_width, image_height = image_size

    # Copy the box
    box = input_box.copy()

    # Convert width and height to x1 and y1
    box[2] += box[0]
    box[3] += box[1]

    # Convert the ratios to pixels
    box[0] *= image_width
    box[1] *= image_height
    box[2] *= image_width
    box[3] *= image_height

    # Convert the floats to ints
    box = [int(x) for x in box]

    return box

def _convert_pil_to_coco(input_box, image_size):
    '''
    Convert a bounding box from PIL format to Coco format.

    Args:
        box: The bounding box to convert.
        image_size: The size of the image.

    Returns:
        The bounding box in Coco format.

    Notes:
        Coco format is [x, y, w, h], ratio of image size.
        PIL format is [x0, y0, x1, y1], pixels.
    '''

    # Get the image size
    image_width, image_height = image_size

    # Copy the box
    box = input_box.copy()

    # Convert x1 and y1 to width and height
    box[2] = box[2] - box[0]
    box[3] = box[3] - box[1]

    # Convert the box to floats
    box = [float(x) for x in box]

    # Convert the ratios to pixels
    box[0] /= image_width
    box[1] /= image_height
    box[2] /= image_width
    box[3] /= image_height

    return box

def _convert_pil_to_selective_search(input_box):
    '''
    Convert a bounding box from PIL format to selective search format.

    Args:
        box: The bounding box to convert.

    Returns:
        The bounding box in selective search format.

    Notes:
        Selective search format is [x, y, w, h], pixels.
        PIL format is [x0, y0, x1, y1], pixels.
    '''

    # Copy the box
    box = input_box.copy()

    # Convert the box to selective search format
    box[2] = box[2] - box[0]
    box[3] = box[3] - box[1]

    return box

def _convert_selective_search_to_pil(input_box):
    '''
    Convert a bounding box from selective search format to PIL format.

    Args:
        box: The bounding box to convert.

    Returns:
        The bounding box in PIL format.

    Notes:
        Selective search format is [x, y, w, h], pixels.
        PIL format is [x0, y0, x1, y1], pixels.
    '''

    # Copy the box
    box = input_box.copy()

    # Convert the box to PIL format
    box[2] = box[2] + box[0]
    box[3] = box[3] + box[1]

    return box

# Write some code that tests the functions above on an example image
# The path to the image is "/home/ec2-user/Documents/Repos/Deep-Learning-in-Computer-Vision/Output.jpg"
if __name__ == "__main__":
    import fiftyone.zoo as foz
    import fiftyone as fo

    from example import Example

    # Load an image using fiftyone and crop it with one of the bounding boxes
    dataset = foz.load_zoo_dataset("coco-2017", split="validation", max_samples=1, label_types = ["segmentations"])

    session = fo.launch_app(dataset)

    # Get the image
    example = dataset.first()
    example = Example(example)

    # Crop the image using the bounding box and PIL
    image = example.get_image()
    bounds = example.get_bounding_box_for(0)

    converted_bounds = convert_box_type(bounds, "coco", "pil", image_size=image.size)
    
    print("Image size:", image.size)

    print("Coco bounds:", bounds)
    print("PIL bounds:", converted_bounds)

    print("Label:", example.get_label_for(0))

    print("Mask shape:", example.get_mask_for(0).shape)

    image = image.crop(converted_bounds)

    # Save the image
    image.save("PIL.jpg")

    # Close the session
    session.wait()