'''
This script counts the amount of examples in a dataset.

It does this by counting the amount of images in the train and validation splits of the dataset.

This is useful for determining the amount of background images to generate for a dataset.

Usage:
    python count_examples.py

Output:
    The amount of examples in the dataset.

Functions:
    count_examples(path_to_dataset: str) -> int
        Counts the amount of examples in a dataset.
'''

import tensorflow as tf
import os

def count_examples(path_to_dataset: str):
    ds_train = tf.keras.utils.image_dataset_from_directory(os.path.join(path_to_dataset, "train"), batch_size=1)
    ds_validation = tf.keras.utils.image_dataset_from_directory(os.path.join(path_to_dataset, "validation"), batch_size=1)

    return len(ds_train) + len(ds_validation)

path = "/home/ec2-user/Documents/datasets/foods"
amount_of_examples = count_examples(path)
print(amount_of_examples)