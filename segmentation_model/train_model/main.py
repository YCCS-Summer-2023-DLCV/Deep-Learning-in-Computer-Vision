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
import matplotlib.pyplot as plt

from segmentation_model.load_dataset.load_dataset import load_dataset
from segmentation_model.train_model.utils import display, normalize_example

path_to_ds = "/home/ec2-user/Documents/datasets/segmentation-dataset"

train_ds = load_dataset(path_to_ds, "train", shuffle = True)
val_ds = load_dataset(path_to_ds, "validation")

for image, label in train_ds.take(1):
    display((image, label))

# normalize the examples
train_ds = train_ds.map(normalize_example, num_parallel_calls = tf.data.AUTOTUNE)
val_ds = val_ds.map(normalize_example, num_parallel_calls = tf.data.AUTOTUNE)

# Create augmentation layers
class AugmentLayer(tf.keras.layers.Layer):
    def __init__(self, seed = 42):
        super().__init__()

        # Both layers should have the same seed so they augment in tandem
        self.augment_inputs = tf.keras.layers.RandomFlip(mode = "horizontal", seed = seed)
        self.augment_mask = tf.keras.layers.RandomFlip(mode = "horizontal", seed = seed)

    def call(self, inputs, mask):
        inputs = self.augment_inputs(inputs)
        mask = self.augment_mask(mask)

        return inputs, mask
    
train_ds = train_ds.cache.map(AugmentLayer).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache.map(AugmentLayer).prefetch(buffer_size = tf.data.AUTOTUNE)

# Define the model
base_model = tf.keras.applications.MobileNetV2(input_shape = [128, 128, 3], include_top = False)

layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project',
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]