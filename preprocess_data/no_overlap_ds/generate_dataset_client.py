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

import tensorflow_datasets as tfds

from preprocess_data.generate_selective_search_examples import generate_selective_search_examples

ds_train, info = tfds.load("coco/2017", split = "train", with_info = True)
all_labels = info.features["objects"]["label"].names
ds_val = tfds.load("coco/2017", split = "validation")

labels = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]

generate_selective_search_examples(ds_train, labels, all_labels, "foods-ss-no-overlap", "train", avoid_overlap_mode = "single_class_examples")
generate_selective_search_examples(ds_val, labels, all_labels, "foods-ss-no-overlap", "validation", avoid_overlap_mode = "single_class_examples")