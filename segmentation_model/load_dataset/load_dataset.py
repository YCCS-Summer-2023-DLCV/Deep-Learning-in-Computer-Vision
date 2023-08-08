'''
This file contains functions to load segmentation datasets.

Author: Tuvya Macklin

Date: 07/17/2023

Version: 1.0.0

Functions:
    `load_dataset(path_to_ds, split, IMG_SIZE, shuffle)` -> `tf.data.Dataset`
        Loads the dataset from the given path and split.

    `load_multiple_datasets(dataset_paths, split, IMG_SIZE, shuffle)` -> `tf.data.Dataset`
        Loads multiple datasets and combines them into one.
'''

import os
import tensorflow as tf
import matplotlib.pyplot as plt

def load_dataset(path_to_ds, split, IMG_SIZE = [128, 128], shuffle = False) -> tf.data.Dataset:
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
    def _decode_image(image, channels, is_masks = False):
        image = tf.io.decode_jpeg(image, channels = channels)

        image = tf.image.resize(image, IMG_SIZE)

        # Convert the mask to a binary mask
        if is_masks:
            # Scale the image to be between 0 and 1
            image = tf.cast(image, tf.float32) / 255.0

            # Round the values to either 0 or 1
            image = tf.round(image)

        return image

    def _process_path(channels = 0, is_masks = False):
        def process(file_path):
            image = tf.io.read_file(file_path)

            image = _decode_image(image, channels = channels, is_masks = is_masks)

            return image

        return process

    # Load the dataset as a bunch of file paths
    path_to_images = os.path.join(path_to_ds, split, "*.jpeg")
    path_to_masks = os.path.join(path_to_ds, split + "-anno", "*.jpeg")

    images = tf.data.Dataset.list_files(path_to_images, shuffle = False)
    masks = tf.data.Dataset.list_files(path_to_masks, shuffle = False)

    # Turn the file paths into images
    images = images.map(_process_path(3), num_parallel_calls = tf.data.AUTOTUNE)
    masks = masks.map(_process_path(1, is_masks = True), num_parallel_calls = tf.data.AUTOTUNE)

    # Zip the images and masks together
    dataset = tf.data.Dataset.zip((images, masks))

    # Shuffle the dataset if necessary
    if shuffle:
        dataset = dataset.shuffle(1000)

    return dataset


def load_multiple_datasets(dataset_paths: list[str], split: str, IMG_SIZE = [128, 128], shuffle: bool = False) -> tf.data.Dataset:
    '''
    Loads multiple datasets and combines them into one.

    Parameters:
        dataset_paths (list[str]): List containing the paths to the datasets.
        split (str): Split to load. Can be either "train" or "val".
        IMG_SIZE (list): Size to resize the images to. Defaults to [128, 128].
        shuffle (bool): Whether or not to shuffle the dataset. Defaults to False.

    Returns:
        dataset (tf.data.Dataset): Dataset containing the images and masks.

     Note:
        The datasets are first loaded individually with the `load_dataset` function.
        Then the masks are scaled by the index of the dataset in the `dataset_paths`
        list. The datasets are then combined and returned.
    '''

    # Load each dataset
    datasets = [load_dataset(path, split, IMG_SIZE, shuffle) for path in dataset_paths]

    # Scale the masks of each dataset. One should not be scaled. The second scaled by 2. The third by 3...
    datasets = [_scale_dataset_masks(dataset, index + 1) for index, dataset in enumerate(datasets)]

    # Combine the dataset with choose_from_datasets
    choice = tf.data.Dataset.range(len(datasets)).repeat()
    main_dataset = tf.data.experimental.choose_from_datasets(datasets, choice)

    # Shuffle the dataset if necessary
    if shuffle:
        main_dataset = main_dataset.shuffle(1000)

    return main_dataset

def _scale_dataset_masks(dataset, scale_factor: int) -> tf.data.Dataset:
    '''
    Scales the masks of the dataset by the given scale factor.

    Parameters:
        dataset (tf.data.Dataset): Dataset to scale.
        scale_factor (int): Factor to scale the masks by.

    Returns:
        dataset (tf.data.Dataset): Dataset with the scaled masks.

    Note:
        The masks are scaled by the scale_factor. The scale_factor must be an integer
        greater than 0. If the scale_factor is 1, nothing is done to the dataset.

        The dataset must be a tf.data.Dataset object. The dataset must be a bunch of
        tuples containing the image and mask. The mask must be the second element in
        the tuple. The mask must be a tensor with a shape of [height, width, 1].
    '''

    # Make sure that the scale_factor is an integer greater than 0
    if not isinstance(scale_factor, int):
        raise TypeError("scale_factor must be an integer")
    
    if scale_factor < 1:
        raise ValueError("scale_factor must be greater than 0")
    
    # If the scale factor is 1, nothing needs to be done
    if scale_factor == 1:
        return dataset
    
    def scale_mask(image, mask):
        # Scale the mask
        return image, tf.math.multiply(mask, scale_factor)

    dataset = dataset.map(scale_mask)

    return dataset


def _two_datasets():

    paths = [
        "/home/ec2-user/Documents/datasets/broccoli-segmentation-dataset",
        "/home/ec2-user/Documents/datasets/banana-segmentation-dataset"
    ]

    dataset = load_multiple_datasets(paths, "validation", shuffle = True)

    # Plot 5 image and masks - you should be able to see the values of the mask pixels
    plt.figure(figsize = (10, 10))
    for index, (image, mask) in enumerate(dataset.take(5)):
        plt.subplot(5, 2, index * 2 + 1)
        plt.imshow(image.numpy().astype("uint8"))

        # plot the mask and show the values of the pixels
        plt.subplot(5, 2, index * 2 + 2)
        plt.imshow(mask.numpy())
        plt.colorbar()

    plt.savefig("plot.png")

def _basic_usage():
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


if __name__ == "__main__":
    _two_datasets()