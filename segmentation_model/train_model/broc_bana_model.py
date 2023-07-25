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
from tensorflow_examples.models.pix2pix import pix2pix

from segmentation_model.load_dataset.load_dataset import load_multiple_datasets
import segmentation_model.train_model.utils as utils

paths = [
    "/home/ec2-user/Documents/datasets/broccoli-segmentation-dataset",
    "/home/ec2-user/Documents/datasets/banana-segmentation-dataset"
]

train_ds = load_multiple_datasets(paths, "train", shuffle = True)
val_ds = load_multiple_datasets(paths, "validation")

for image, label in train_ds.take(1):
    utils.display((image, label))

# normalize the examples
train_ds = train_ds.map(utils.normalize_example, num_parallel_calls = tf.data.AUTOTUNE)
val_ds = val_ds.map(utils.normalize_example, num_parallel_calls = tf.data.AUTOTUNE)

# Create augmentation layers
BATCH_SIZE = 64
train_ds = train_ds.batch(BATCH_SIZE).map(utils.AugmentLayer()).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).map(utils.AugmentLayer()).prefetch(buffer_size = tf.data.AUTOTUNE)

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

down_stack = tf.keras.Model(inputs = base_model.input, outputs = base_model_outputs)
down_stack.trainable = True

# Build the upstack which goes from 4x4 -> 8x8 -> ... -> 64x64
up_stack = [
    pix2pix.upsample(512, 3), 
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),
]

model = utils.get_unet_model(3, down_stack, up_stack)

LEARNING_RATE = 0.001 # default is 0.001

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    # Use IOU metric
    metrics = [tf.keras.metrics.IoU(num_classes = 3, target_class_ids=[1, 2], sparse_y_pred = False)]
)

model_name = "both_100"

tensorboard_callback = utils.get_tensorboard_callback(model_name)

EPOCHS = 100
history = model.fit(
    train_ds,
    epochs = EPOCHS,
    validation_data = val_ds,
    callbacks = [tensorboard_callback]
)

utils.show_predictions(val_ds, model, 1)

utils.save_model(model, model_name)

utils.plot_history(history, model_name, aspects = ["io_u"])