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

        The structure of the dataset should be as follows:
        ```
            dataset
            ├── train
            │   ├── image_1
            │   ├── image_2
            │   ├── ...
            │   └── image_n
            └── train-anno
            │   ├── mask_1
            │   ├── mask_2
            │   ├── ...
            │   └── mask_n
            ├── val
            │   ├── image_1
            │   ├── image_2
            │   ├── ...
            │   └── image_n
            └── val-anno
                ├── mask_1
                ├── mask_2
                ├── ...
                └── mask_n
        ```



    '''

    # Define the functions to process the paths
    def _decode_image(image, channels):
        image = tf.io.decode_jpeg(image, channels = channels)

        return tf.image.resize(image, IMG_SIZE)

    def _process_path(channels = 0):
        def process(file_path):
            image = tf.io.read_file(file_path)

            image = _decode_image(image, channels = channels)

            return image

        return process

    # Load the dataset as a bunch of file paths
    path_to_images = os.path.join(path_to_ds, split, "*.jpeg")
    path_to_masks = os.path.join(path_to_ds, split + "-anno", "*.jpeg")

    images = tf.data.Dataset.list_files(path_to_images, shuffle = False)
    masks = tf.data.Dataset.list_files(path_to_masks, shuffle = False)

    # Turn the file paths into images
    images = images.map(_process_path(3), num_parallel_calls = tf.data.AUTOTUNE)
    masks = masks.map(_process_path(1), num_parallel_calls = tf.data.AUTOTUNE)

    # Zip the images and masks together
    dataset = tf.data.Dataset.zip((images, masks))

    # Shuffle the dataset if necessary
    if shuffle:
        dataset = dataset.shuffle(1000)

    return dataset

def load_multiple_datasets(paths_and_labels: dict[str], split: str, IMG_SIZE = [128, 128], shuffle: bool = False):
    '''
    Loads multiple datasets and combines them into one.

    Parameters:
        paths_and_labels (dict[str]): Dictionary containing the paths to the datasets and the labels for each dataset.
        split (str): Split to load. Can be either "train" or "val".
        IMG_SIZE (list): Size to resize the images to. Defaults to [128, 128].
        shuffle (bool): Whether or not to shuffle the dataset. Defaults to False.

    Returns:
        dataset (tf.data.Dataset): Dataset containing the images and masks.

    Note:
        The structure of the dictionary should be as follows:
        ```python
            paths_and_labels = {
                "path_to_ds_1": "label_1",
                "path_to_ds_2": "label_2",
                ...
                "path_to_ds_n": "label_n"
            }
        ```
    '''
    # Load each dataset

    # Scale the masks of each dataset. One should not be scaled. The second scaled by 2. The third by 3...

    # Combine the datasets
    pass


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