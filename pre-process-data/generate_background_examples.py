import tensorflow_datasets as tfds
import tensorflow as tf
import random
import os
from PIL import Image

from process_data_api import _get_label_ids

PATH_TO_DATASET = "/home/ec2-user/Documents/datasets/foods"

def generate_background_examples(n, source_dataset, label_ids_to_avoid):
    # Create directories in the food dataset for the background examples
    _create_background_directories()

    examples_generated = 0
    
    while examples_generated < n:
        for example in source_dataset:
            if examples_generated >= n:
                return

            if not _contains_label_in_subset(example, label_ids_to_avoid):
                _crop_and_save_random_patch(example, examples_generated, n)
                examples_generated += 1

def _create_background_directories():
    for split in ["train", "validation"]:
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

def _crop_and_save_random_patch(example, example_index, total_examples, train_val_split = 0.95):
    # Generate the path for the image.
    # If example_index is within the percentage of examples for the training split, save it there
    split = "train"
    if example_index / float(total_examples) > train_val_split:
        split = "validation"
    
    # Name the file with the image id and exmample index
    file_name = str(example["image/id"].numpy()) + "_" + str(example_index) + ".jpeg"

    path_to_image = os.path.join(PATH_TO_DATASET, split, "background", file_name)

    # Save the image and load it with PIL
    tf.keras.preprocessing.image.save_img(path_to_image, example["image"])
    img = Image.open(path_to_image)

    # Generate random bounds with which to crop the image
    # PIL expects the bounds to be [x0, y0, x1, y1]
    bounds = [random.random(), random.random()]
    bounds += [random.uniform(bounds[0], 1), random.uniform(bounds[1], 1)]

    # Scale the bounds to the size of the image and convert them to ints
    bounds = [int(bound * img.size[i % 2]) for i, bound in enumerate(bounds)]
    
    # Ensure that the bounds are not too small
    if bounds[2] - bounds[0] < 10:
        bounds[2] = bounds[0] + 10
    if bounds[3] - bounds[1] < 10:
        bounds[3] = bounds[1] + 10
    
    # Ensure that the bounds are not too large
    if bounds[2] - bounds[0] > img.size[0]:
        bounds[2] = bounds[0] + img.size[0]
    if bounds[3] - bounds[1] > img.size[1]:
        bounds[3] = bounds[1] + img.size[1]

    img = img.crop(bounds)
    img.save(path_to_image)
    

dataset, info= tfds.load("coco/2017", split = "train", with_info = True)
all_labels = info.features["objects"]["label"].names

amount_of_examples = 66347
background_factor = 5

example_labels = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]
example_label_ids = _get_label_ids(example_labels, all_labels)

generate_background_examples(amount_of_examples * background_factor, dataset, example_label_ids)