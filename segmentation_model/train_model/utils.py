'''
This file contains utility functions for training the model

Functions:
    `display(display_list, to_file, root_dir, file_name, count)`
        Display a list of images and their masks
    `ensure_directory_exists(dir)`
        Ensure that a directory exists

Author: Tuvya Macklin

Date: 07/17/2023

Version: 1.0.0

'''

import matplotlib.pyplot as plt
import tensorflow as tf
import os


# Display an example with its mask
def display(display_list, to_file = True, root_dir = "segmentation_model/train_model/plots", file_name = "img_and_mask", count = None):
    '''
    Display a list of images and their masks

    Args:
        display_list (list): A list of images and their masks. The image and mask should be numpy arrays.
        to_file (bool): Whether to save the image to a file or not
        root_dir (str): The root directory to save the image to
        file_name (str): The name of the file to save the image to
        count (int): The number of the image to save
    
    Returns:
        None

    Side Effects:
        Saves the image to a file
    '''

    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    
    if to_file:
        path = os.path.join(root_dir, file_name)
        if not count is None:
            path += "-" + str(count)

        path += ".png"
        plt.savefig(path)
    else:
        plt.show()

def ensure_directory_exists(dir):
    '''
    Ensure that a directory exists

    Args:
        dir (str): The directory to ensure exists

    Returns:
        None

    Side Effects:
        Creates the directory if it does not exist
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)

def normalize_example(image, mask):
    '''
    Normalize an image

    Args:
        image (tf.Tensor): The image to normalize
        mask (tf.Tensor): The mask to normalize
        
    Returns:
        image (tf.Tensor): The normalized image
        mask (tf.Tensor): The normalized mask
    '''

    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask