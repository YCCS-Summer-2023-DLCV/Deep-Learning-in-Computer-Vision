import os

import fiftyone.zoo as foz
from tqdm import tqdm
from PIL import Image

from process_example import process_example

ROOT_DIR = "/home/ec2-user/Documents/datasets"

def generate_dataset(ds_name, split, classes, min_size = None):
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split = split,
        classes = classes,
        label_types = ["segmentations"]
    )

    directories = _create_directories(ds_name, split, classes)

    for example in tqdm(dataset):
        datapoints = process_example(example, classes, use_selective_search = False)
        for datapoint in datapoints:
            if _datapoint_is_big_enough(datapoint, min_size):
                _save_datapoint(datapoint, directories)

def _datapoint_is_big_enough(datapoint, min_size):
    # Check if min_size is specified
    if min_size is None:
        return True
    
    # Check if the image is big enough
    image = datapoint["image"]
    width, height = image.size

    if width < min_size[0] or height < min_size[1]:
        return False
    
    return True

def _save_datapoint(datapoint, directories):
    # Convert the image and mask from numpy arrays to images
    image = datapoint["image"]
    mask = Image.fromarray(datapoint["mask"])

    # Create the path for each file
    example_path = directories[0]
    mask_path = directories[1]
    
    example_path = os.path.join(example_path, datapoint["id"] + ".jpeg")
    mask_path = os.path.join(mask_path, datapoint["id"] + ".jpeg")

    # Save each one
    try:
        image.save(example_path)
        mask.save(mask_path)
    except Exception as error:
        with open("errors.txt", "a") as file:
            file.write("Failed to save image or mask of" + datapoint["id"] + "\n")
            file.write(str(error) + "\n")
            file.write("This was the attempted path: " + example_path)

def _create_directories(ds_name, split, classes):
    split_dir = os.path.join(ROOT_DIR, ds_name, split)
    anno_dir = os.path.join(ROOT_DIR, ds_name, str(split) + "-anno")

    def _ensure_directory_exists(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    _ensure_directory_exists(split_dir)
    _ensure_directory_exists(anno_dir)

    return split_dir, anno_dir
    

if __name__ == "__main__":
    classes = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]

    generate_dataset(ds_name = "segmentation-dataset-test", split = "train", classes = classes)
    generate_dataset(ds_name = "segmentation-dataset", split = "validation", classes = classes)