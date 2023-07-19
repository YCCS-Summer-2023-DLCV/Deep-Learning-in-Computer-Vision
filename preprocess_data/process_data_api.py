import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from PIL import Image

ROOT_DIR = "/home/ec2-user/Documents/datasets"

def process_data_from_class_subset(label_subset, dataset_title):
    ds_train, info = tfds.load("coco/2017", split = "train", shuffle_files = True, with_info= True)
    ds_validation = tfds.load("coco/2017", split = "validation", shuffle_files = True)

    all_labels = info.features["objects"]["label"].names
    subset_ids = _get_label_ids(label_subset, all_labels)

    label_subset_registry = {id: name for id, name in zip(subset_ids, label_subset)}
    ds_dir = os.path.join(ROOT_DIR, dataset_title)

    # Filter out examples that don't have a label in the label subset

    _preprocess_data(ds_train, label_subset_registry, ds_dir, "train")
    _preprocess_data(ds_validation, label_subset_registry, ds_dir, "validation")

def _preprocess_data(dataset, label_subset_registry, ds_dir, split_name):
    # Create a directory to hold this split and a directory in it to hold each class
    split_directory = _create_directories(ds_dir, split_name, label_subset_registry.values())

    # Filter out any images that are labeled with something in the subset
    # Crop them to only include the object they are labeled with
    # Save them

    for example in tqdm(dataset, desc = "Preparing the " + split_name + " split"):
        _crop_and_save(example, split_directory, label_subset_registry)
    
    plt.savefig("examples.png")

def _get_label_ids(user_labels, all_labels):
    id_subset = []

    for user_label in user_labels:
        for index, coco_label in enumerate(all_labels):
            if user_label == coco_label:
                id_subset.append(index)
    
    return id_subset

def _create_directories(ds_dir, split_name, label_subset):
    def ensure_directory_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    split_dir = os.path.join(ds_dir, split_name)

    for label in label_subset:
        ensure_directory_exists(os.path.join(split_dir, label))
    
    return split_dir

def _crop_and_save(example, split_directory, label_subset_registry):
    # Store the bounding box of every object in the image that is labeled with something in the subset
    labels = example["objects"]["label"].numpy()
    food_objects = []
    for label_index, label_id in enumerate(labels):
        if label_id in label_subset_registry.keys():
            image_class = label_subset_registry[label_id]
            food_objects.append((label_index, image_class))
    
    # For each bounding box identified, create a cropped image with it and save it
    for index, label in food_objects:
        bounds = _get_bounds_for_crop(example, index)
        path_to_image = _get_path_for_cropped_image(split_directory, label, example, index)
        _save_image_from_bounds(example, bounds, path_to_image)

def _get_path_for_cropped_image(split_directory, label, example, index):
    file_name = str(example["image/id"].numpy()) + "-" + str(index) + ".jpeg"
    return os.path.join(split_directory, label, file_name)

def _save_image_from_bounds(example, bounds, path_to_image):
    # Save the image
    tf.keras.preprocessing.image.save_img(path_to_image, example["image"])

    # Load it with PIL, crop it, and resave it
    img = Image.open(path_to_image)
    img = img.crop(bounds)
    img.save(path_to_image)

def _get_bounds_for_crop(example, index):
    # Coco stores the bounds as [y0, x0, y1, x1]
    # PIL expects the bounds to be [x0, y0, x1, y1]
    bounds = example["objects"]["bbox"][index].numpy()
    bounds[0], bounds[1] = bounds[1], bounds[0]
    bounds[2], bounds[3] = bounds[3], bounds[2]

    # Expand the bounding box a certain percentage to give context
    # If the box goes out of the image, rein it in
    expansion_constant = 0.1
    bounds[0] *= 1 - expansion_constant
    bounds[1] *= 1 - expansion_constant
    bounds[2] *= 1 + expansion_constant
    bounds[3] *= 1 + expansion_constant
    
    if bounds[0] < 0:
        bounds[0] = 0
    if bounds[1] < 0:
        bounds[1] = 0
    if bounds[2] > 1:
        bounds[2] = 1
    if bounds[3] > 1:
        bounds[3] = 1

    # Coco stores the bounds as ratios of the width and height of the image
    # The crop method expects number of pixels
    image_shape = example["image"].shape
    height, width = image_shape[0], image_shape[1]
    bounds[0] *= width
    bounds[2] *= width

    bounds[1] *= height
    bounds[3] *= height

    return bounds