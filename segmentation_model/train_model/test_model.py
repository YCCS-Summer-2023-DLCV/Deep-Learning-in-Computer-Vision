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

import tensorflow as tf

from segmentation_model.load_dataset.load_dataset import load_dataset
from segmentation_model.train_model.utils import show_predictions, normalize_example, load_model

model = load_model("first_model")

path_to_ds = "/home/ec2-user/Documents/datasets/segmentation-dataset"
test_ds = load_dataset(path_to_ds, "validation")

test_ds = test_ds.map(normalize_example, num_parallel_calls = tf.data.AUTOTUNE)

test_ds = test_ds.batch(32)

model.evaluate(test_ds)

show_predictions(test_ds, model, 10)