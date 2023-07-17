'''
This file contains the function to load the dataset.

Author: Tuvya Macklin

Date: 07/17/2023

Version: 1.0.0

Functions:
    `load_dataset(path_to_ds, split, IMG_SIZE, shuffle)` -> `tf.data.Dataset`
        Loads the dataset from the given path and split.
'''

import os
import tensorflow as tf
import matplotlib.pyplot as plt

def load_dataset(path_to_ds, split, IMG_SIZE = [128, 128], shuffle = False):
    '''
    Loads the dataset from the given path and split.

    Parameters:
        path_to_ds (str): Path to the dataset.
        split (str): Split to load. Can be either "train" or "val".
        IMG_SIZE (list): Size to resize the images to. Defaults to [128, 128].
        shuffle (bool): Whether or not to shuffle the dataset. Defaults to False.

    Returns:
        dataset (tf.data.Dataset): Dataset containing the images and masks.

    Note:
        The dataset is loaded as a tf.data.Dataset object. The dataset is
        loaded as a bunch of file paths and then decoded into images. The
        images are then resized to the IMG_SIZE which is [128, 128] by default.
        The dataset is then zipped together with the masks and returned.
    '''

    # Define the functions to process the paths
    def _decode_image(image):
        image = tf.io.decode_jpeg(image)

        return tf.image.resize(image, IMG_SIZE)

    def _process_path(file_path):
        image = tf.io.read_file(file_path)

        image = _decode_image(image)

        return image

    # Load the dataset as a bunch of file paths
    path_to_images = os.path.join(path_to_ds, split, "*.jpeg")
    path_to_masks = os.path.join(path_to_ds, split + "-anno", "*.jpeg")

    images = tf.data.Dataset.list_files(path_to_images, shuffle = False)
    masks = tf.data.Dataset.list_files(path_to_masks, shuffle = False)

    # Turn the file paths into images
    images = images.map(_process_path, num_parallel_calls = tf.data.AUTOTUNE)
    masks = masks.map(_process_path, num_parallel_calls = tf.data.AUTOTUNE)

    # Zip the images and masks together
    dataset = tf.data.Dataset.zip((images, masks))

    # Shuffle the dataset if necessary
    if shuffle:
        dataset = dataset.shuffle(1000)

    return dataset



if __name__ == "__main__":
    path = "/home/ec2-user/Documents/datasets/segmentation-dataset"
    dataset = load_dataset(path, "train")

    for example in dataset.take(1):
        image, mask = example

        plt.figure(figsize = (7, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(image.numpy().astype("uint8"))

        plt.subplot(1, 2, 2)
        plt.imshow(mask.numpy())

        plt.savefig("plot.png")