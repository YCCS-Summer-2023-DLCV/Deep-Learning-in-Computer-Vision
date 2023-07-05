import tensorflow_datasets as tfds
import tensorflow as tf
import random
import os
from PIL import Image

from process_data_api import _get_label_ids

PATH_TO_DATASET = "/home/ec2-user/Documents/datasets/foods-ss"

def _set_path():
    # Add the root of the venv to the path
    # This is necessary to import the selective_search module

    # Get the path to the venv
    venv_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Add the venv path to the path
    import sys
    sys.path.append(venv_path)

_set_path()

from selective_search import selective_search_fast
from generate_selective_search_examples import _crop_and_save_image

def generate_background_examples(crops_per_image, source_dataset, label_ids_to_avoid, split):
    # Create directories in the food dataset for the background examples
    _create_background_directories(split)

    for example in source_dataset:
        if _contains_label_in_subset(example, label_ids_to_avoid):
            continue

        boxes = _get_n_boxes_with_selective_search(example, crops_per_image)

        for index, box in enumerate(boxes):
            path_for_image = _get_path_for_cropped_example(example, index)
            _crop_and_save_image(example, box, path_for_image)

# TODO
def _get_n_boxes_with_selective_search(example, n):
    pass

# TODO
def _get_path_for_cropped_example(example, additional_token):
    pass

def _create_background_directories(split):
    split_path = os.path.join(PATH_TO_DATASET, split, "background")
    _ensure_directory_exists(split_path)

def _ensure_directory_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

def _contains_label_in_subset(example, label_subset):
    for label in example["objects"]["label"].numpy():
        if label in label_subset:
            return True
        
    return False

    

train_ds, info= tfds.load("coco/2017", split = "train", with_info = True)
val_ds = tfds.load("coco/2017", split = "validation")
all_labels = info.features["objects"]["label"].names

# We have generated 264,652 labeled examples in the train split of dataset so far.
# We need to generate five (background_factor) times that amount of background images.
# There are 102,032 images in Coco train without our labels. Those will be used to generate the background images.
# These lines determine how many crops should be taken from each image.
# The same logic is used to determine how many crops are taken from each image in Coco validation without our label
background_factor = 5
crops_per_train_example = int((264652 * background_factor) / 102032)
crops_per_validation_example = int((11941 * background_factor) / 4292)

example_labels = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]
example_label_ids = _get_label_ids(example_labels, all_labels)

generate_background_examples(crops_per_train_example, train_ds, example_label_ids, split = "train")
generate_background_examples(crops_per_validation_example, val_ds, example_label_ids, split = "validation")