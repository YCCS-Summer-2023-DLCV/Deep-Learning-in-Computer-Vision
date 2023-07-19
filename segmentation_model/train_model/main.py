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

import matplotlib.pyplot as plt

from segmentation_model.load_dataset.load_dataset import load_dataset
from segmentation_model.train_model.utils import display, normalize_example, show_predictions, save_model, get_tensorboard_callback

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
    
BATCH_SIZE = 32
train_ds = train_ds.batch(BATCH_SIZE).map(AugmentLayer()).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).map(AugmentLayer()).prefetch(buffer_size = tf.data.AUTOTUNE)

# Define the model
base_model = tf.keras.applications.MobileNetV2(input_shape = [128, 128, 3], include_top = False)


resnet_names = [
    "conv2_block1_1_relu",
    "conv3_block1_1_relu",
    "conv4_block1_1_relu",
    "conv5_block1_1_relu",
    "conv5_block3_out",
]

layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project',
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs = base_model.input, outputs = base_model_outputs)
# down_stack.trainable = False

# Build the upstack which goes from 4x4 -> 8x8 -> ... -> 64x64
up_stack = [
    pix2pix.upsample(512, 3), 
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),
]

def get_unet_model(output_channels: int):
    inputs = tf.keras.layers.Input(shape = [128, 128, 3])

    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters = output_channels,
        kernel_size = 3,
        strides = 2,
        padding = "same"
    )

    x = last(x)

    return tf.keras.Model(inputs = inputs, outputs = x)

model = get_unet_model(output_channels = 1)
model.compile(
    optimizer = "adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ["accuracy"]
)

show_predictions(val_ds, model, 1)

model_name = "30_epochs"

tensorboard_callback = get_tensorboard_callback(model_name)

EPOCHS = 30
model.fit(
    train_ds,
    epochs = EPOCHS,
    validation_data = val_ds,
    callbacks = [tensorboard_callback]
)

show_predictions(val_ds, model, 1)

save_model(model, model_name)