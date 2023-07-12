import math
import os
import tempfile

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

from process_data_api import _get_label_ids, ROOT_DIR

def _set_path():
    # Add the root of the venv to the path
    # This is necessary to import the selective_search module

    # Get the path to the venv
    venv_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Add the venv path to the path
    import sys
    sys.path.append(venv_path)

_set_path()

from inference.selective_search import selective_search_fast

SS_EXAMPLES_PER_ITEM = 5

def generate_selective_search_examples(dataset, label_subset, all_labels, dataset_title, split):
    # Get the label registry
    label_ids = _get_label_ids(label_subset, all_labels)
    label_registry = {id: name for id, name in zip(label_ids, label_subset)}

    # Generate the path for the dataset directory
    split_dir = _create_directories(os.path.join(ROOT_DIR, dataset_title), split, label_subset)

    # For each example, crop out patches using selective search boxes that encompass the coco box
    for example in tqdm(dataset):
        _process_example(example, label_registry, split_dir)

def _create_directories(ds_dir, split_name, label_subset):
    def ensure_directory_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    split_dir = os.path.join(ds_dir, split_name)

    for label in label_subset:
        ensure_directory_exists(os.path.join(split_dir, label))
    
    return split_dir

def _process_example(example, label_registry, split_dir):
    # Get a list of items in the example that are in the subset and their corresponding boxes
    # Store them in a list of tuples: (item_label_id, item_bbox)
    items = _get_items_and_boxes_in_example(example, label_registry)

    if len(items) == 0:
        return

    # Get selective search boxes
    ss_boxes = _get_ss_boxes_for_example(example)

    # Convert ss_boxes to coco format
    ss_boxes = _convert_ss_boxes_to_coco(ss_boxes, example["image"].shape)

    # For each item, pick a few selective_search boxes to use as examples
    # The boxes should be a list of tuples. Each tuple is in the following format:
    #   (label name, box)
    boxes = []
    for item_id, bbox in items:
        for ss_box in _pick_good_boxes(ss_boxes, bbox):
            boxes.append([label_registry[item_id], ss_box])

    for index, (_, box) in enumerate(boxes):
        boxes[index][1] = _convert_coco_bounds_to_PIL_bounds(box.copy() , example["image"].shape)

    # For each box, crop the image and save the image
    # The path should be: split_dir/class/<image id>-<crop number>.jpeg
    for index, (label, box) in enumerate(boxes):
        # Generate the path to save the image
        path_to_save = os.path.join(split_dir, label, f"{example['image/id'].numpy()}-{index}.jpeg")
        _crop_and_save_image(example, box, path_to_save)

def _crop_and_save_image(example, box, path_for_image):
    # Coco stores the bounds as [y0, x0, y1, x1]
    # PIL expects the bounds to be [x0, y0, x1, y1]

    temp_file = tempfile.NamedTemporaryFile(suffix = ".jpeg")
    temp_file_name = temp_file.name

    tf.keras.preprocessing.image.save_img(temp_file_name, example["image"])

    img = Image.open(temp_file_name)

    img = img.crop(box)
    
    img.save(path_for_image)

    temp_file.close()

def _get_items_and_boxes_in_example(example, label_registry):
    '''
    Get a list of items in the example that are in the subset and their corresponding boxes
    Store them in a list of tuples: (item_label_id, item_bbox)

    Parameters:
        example: the example to get the items from
        label_registry: a dictionary mapping label ids to label names
    
    Returns:
        A list of tuples: (item_label_id, item_bbox)
    '''

    # Get a list of items in the example that are in the subset and their corresponding boxes

    # Get all the items in the example
    all_item_ids = example["objects"]["label"].numpy()

    # Filter out the items that aren't in the subset
    items_in_subset = []
    for index, item_id in enumerate(all_item_ids):
        if item_id in label_registry.keys():
            # Get the bounding box for the item
            bbox = example["objects"]["bbox"][index].numpy()
            # Add the item and its bounding box to the list
            items_in_subset.append((item_id, bbox))
    
    return items_in_subset

def _get_ss_boxes_for_example(example):
    temp_file = tempfile.NamedTemporaryFile(suffix = ".jpeg")
    temp_file_name = temp_file.name

    tf.keras.preprocessing.image.save_img(temp_file_name, example["image"])

    recs = selective_search_fast(temp_file_name)

    temp_file.close()
    return recs

def _pick_good_boxes(ss_boxes, bbox):
    good_boxes = []
    # Iterate through the ss_boxes
        # If the ss_box encompasses the bbox, AND is within 0.3 of the bounds, track it
    for index, ss_box in enumerate(ss_boxes):
        if _is_good_box(ss_box, bbox):
            good_boxes.append(ss_box)
            np.delete(ss_boxes, index, 0)

        if len(good_boxes) >= 5:
            return good_boxes
            
    return good_boxes

def _convert_ss_boxes_to_coco(ss_boxes, image_shape):
    # Convert the ss_box to the same format as Coco
    # This is the selective search format(x, y, w, h). It is stored in pixel values
    # Coco stores the bounds as [y0, x0, y1, x1] and as a ratio of the image size

    # Get the image size
    image_height, image_width, _ = image_shape

    # Convert the array as one that holds floats
    ss_boxes = np.asfarray(ss_boxes)

    for box in ss_boxes:
        # Convert w, h to x2, y2
        box[2] += box[0]
        box[3] += box[1]

        # Scale the pixel values to floats between 0 and 1
        box[0] /= float(image_width)
        box[2] /= float(image_width)
        box[1] /= float(image_height)
        box[3] /= float(image_height)

        # Swap the x values and y values
        box[0], box[1] = box[1], box[0]
        box[2], box[3] = box[3], box[2]

    return ss_boxes

def _convert_coco_bounds_to_PIL_bounds(bounds, image_shape):
    # Coco stores the bounds as [y0, x0, y1, x1]
    # PIL expects the bounds to be [x0, y0, x1, y1]
    bounds[0], bounds[1] = bounds[1], bounds[0]
    bounds[2], bounds[3] = bounds[3], bounds[2]

    # Coco stores the bounds as ratios of the width and height of the image
    # The crop method expects number of pixels
    height, width, _ = image_shape
    bounds[0] *= width
    bounds[2] *= width

    bounds[1] *= height
    bounds[3] *= height

    return bounds

def _is_good_box(ss_box, bbox):
    # If the ss_box encompasses the bbox, AND is within 0.3 of the bounds, return True
    # Otherwise, return False
    # The format of the boxes is the Coco format: [y0, x0, y1, x1]
    #TODO

    EXPANSION_CONSTANT = 0.3

    if not (ss_box[0] < bbox[0] and ss_box[1] < bbox[1]):
        return False
    
    if not (ss_box[2] > bbox[2] and ss_box[3] > bbox[3]):
        return False
    
    dist_between_top_left = math.dist(ss_box[0:2], bbox[0:2])
    dist_between_bottom_right = math.dist(ss_box[2:4], bbox[2:4])

    if dist_between_top_left > EXPANSION_CONSTANT or dist_between_bottom_right > EXPANSION_CONSTANT:
        return False

    return True

# Coco stores the bounds as [y0, x0, y1, x1] and as a ratio of the image size
# This is the selective search format(x, y, w, h). It is stored in pixel values