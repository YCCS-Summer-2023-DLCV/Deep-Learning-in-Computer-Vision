import os

import fiftyone.zoo as foz
from PIL import Image

from process_example import process_example

ROOT_DIR = "/home/ec2-user/Documents/datasets"

def generate_dataset(ds_name, split, classes):
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split = split,
        classes = classes,
        label_types = ["segmentations"],
        max_samples = 10
    )

    split_dir = _create_directories(ds_name, split, classes)

    for example in dataset:
        datapoints = process_example(example, classes, use_selective_search = False)
        for datapoint in datapoints:
            _save_datapoint(datapoint, split_dir)

def _save_datapoint(datapoint, split_dir):
    # Convert the image and mask from numpy arrays to images
    image = datapoint["image"]
    mask = Image.fromarray(datapoint["mask"])

    # Create the path for each file
    base_path = os.path.join(split_dir, datapoint["label"], datapoint["id"])
    image_path = base_path + ".jpeg"
    mask_path = base_path + "-mask" + "..jpeg"

    # Save each one
    image.save(image_path)
    mask.save(mask_path)

def _create_directories(ds_name, split, classes):
    split_dir = os.path.join(ROOT_DIR, ds_name, split)

    def _ensure_directory_exists(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    for label in classes:
        label_dir = os.path.join(split_dir, label)
        _ensure_directory_exists(label_dir)

    return split_dir
    

if __name__ == "__main__":
    classes = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]

    generate_dataset(ds_name = "segmentation-dataset", split = "train", classes = classes)
    generate_dataset(ds_name = "segmentation-dataset", split = "validation", classes = classes)