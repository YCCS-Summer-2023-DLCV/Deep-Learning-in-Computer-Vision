'''
This file contains utility functions for training the model

Functions:
    `display(display_list, to_file, root_dir, file_name, count)`
        Display a list of images and their masks
    `ensure_directory_exists(dir)`
        Ensure that a directory exists
    `normalize_example(image, mask)`
        Normalize an image and its mask
    `show_predictions(dataset, model, num)`
        Show predictions for a dataset
    `create_mask(pred_mask)`
        Create a mask from a prediction
    `save_model(model, model_name)`
        Save a model
    `load_model(model_name)`
        Load a model
    `get_tensorboard_callback(model_name)`
        Get a TensorBoard callback for a model
    `plot_history(history, model_name, aspects, height, length, to_file, root_dir)`
        Plot the history of a model

Author: Tuvya Macklin

Date: 07/17/2023

Version: 1.0.0

'''

import matplotlib.pyplot as plt
import tensorflow as tf
import os
import datetime

# Display an example with its mask

def display(display_list, to_file = True, root_dir = ".tuvya_stuff/plots/display", file_name = "img_and_mask", count = None, color_bar = False):
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

    ensure_directory_exists(root_dir)

    plt.figure(figsize=(10, 10 * len(display_list)))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i].numpy())
        plt.axis("off")

        if i > 0 and color_bar:
            plt.colorbar()
    
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

    Notes:
        The image and mask are normalized by dividing by 255.0
    '''

    image = tf.cast(image, tf.float32) / 255.0

    return image, mask

def create_mask(prediction):
    # Get the maximum value along the last axis
    # To elaborate:
    #   The prediction tensor has shape (batch_size, height, width, num_classes)
    #   The last axis is the axis with the num_classes
    #   The maximum value along this axis is the predicted class
    #   The predicted class is the mask
    prediction = tf.argmax(prediction, axis = -1)
    prediction = prediction[..., tf.newaxis]

    return prediction[0]


def show_predictions(dataset, model, num = None, root_dir = ".tuvya_stuff/plots/predictions", default_file_name = "prediction", color_bar = False):
    '''
    Display a list of images and their masks

    Args:
        dataset (tf.data.Dataset): The dataset to get the images and masks from
        num (int): The number of images to display
        model (tf.keras.Model): The model to use to predict the masks
        root_dir (str): The root directory to save the image to
        file_name (str): The name of the file to save the image to
    
    Returns:
        None

    Side Effects:
        Saves the image to a file
    '''

    multiple = True
    if num is None:
        num = 1
        multiple = False

    for index, (image, mask) in enumerate(dataset.take(num)):
        pred_mask = model.predict(image)

        file_name = default_file_name
        if multiple:
            file_name = default_file_name + "-" + str(index)


        display(
            [image[0], mask[0], create_mask(pred_mask)],
            root_dir = root_dir,
            file_name = file_name,
            color_bar = color_bar
        )

def save_model(model, model_name, root_model_dir = ".tuvya_stuff/models"):
    '''
    Saves a model to the given directory.

    Parameters:
        model (tf.keras.Model): The model to save.
        model_name (str): The name of the model.
        root_model_dir (str): The root directory for the models.
    
    Raises:
        FileNotFoundError: If the model directory doesn't exist.
    
    Returns:
        None
    
    Notes:
        The model is saved to root_model_dir/model_name/model.keras.
        The root_model_dir is "segmentation_model/models" by default.
    '''
    # Ensure that a directory exists for the model
    # The path should be model_dir/model_name
    ensure_directory_exists(os.path.join(root_model_dir, model_name))

    # Save the model
    model.save(os.path.join(root_model_dir, model_name, "model.keras"))

def load_model(model_name, root_model_dir = ".tuvya_stuff/models", path_to_model = None):
    '''
    Loads a model from the given directory.

    Parameters:
        model_name (str): The name of the model.
        root_model_dir (str): The root directory for the models.
        path_to_model (str): The path to the model file. If None, the path is root_model_dir/model_name/model.keras.
    
    Returns:
        The loaded model.
    
    Raises:
        FileNotFoundError: If the model file doesn't exist.
    
    Notes:
        The model is loaded from root_model_dir/model_name/model.keras.
        The root_model_dir is "segmentation_model/models" by default.
    '''
    # Ensure that the model file exists
    path = None
    if path_to_model is None:
        path = os.path.join(root_model_dir, model_name, "model.keras")
    else:
        path = path_to_model
   
    if not os.path.isfile(path):
        raise FileNotFoundError

    # Load the model
    model = tf.keras.models.load_model(path)
    return model

def get_tensorboard_callback(model_name: str, root_dir = ".tuvya_stuff/tensorboard"):
    '''
    Returns a TensorBoard callback for a model with the given name.

    Parameters:
        model_name (str): The name of the model.
    
    Returns:
        A TensorBoard callback.
    
    Notes:
        The TensorBoard callback is saved to root_tensorboard_dir/fit/model_name/current_date_and_time.

        To view the TensorBoard, run the following command in the terminal:
        ```
        tensorboard --logdir root_tensorboard_dir/fit/model_name
        ```
        `root_tensorboard_dir` is the root directory for the TensorBoard callbacks. The default is .tensorboard.
    '''
    log_dir = os.path.join(root_dir, "fit", model_name, datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback

def plot_history(history, model_name = None, aspects = ["accuracy"], height = 7, length = 7, to_file = True, root_dir = ".tuvya_stuff/plots/history"):
    '''
    Plots the history of a model.

    Parameters:
        history (tf.keras.callbacks.History): The history of the model.
        model_name (str): The name of the model.
        aspects (list): The aspects of the model to plot. The default is ["accuracy"].
        height (int): The height of the plot. The default is 7.
        length (int): The length of the plot. The default is 7.
        to_file (bool): Whether to save the plot to a file or not. The default is True.
        root_dir (str): The root directory to save the plot to. The default is "segmentation_model/train_model/plots/history".

    Returns:
        None

    Side Effects:
        Saves the plot to a file.

    Notes:
        The plot is saved to root_dir/history-model_name.png if to_file is True.
        If no model name is given, the plot is saved to root_dir/history-HH:MM.png, where HH:MM is the current time.
    '''
    
    plt.figure(figsize = (height, length))

    for count, aspect in enumerate(aspects):
        plt.subplot(1, len(aspects), count + 1)
        plt.plot(history.history[aspect], label = "Training " + aspect)
        plt.plot(history.history["val_" + aspect], label = "Validation " + aspect)

        plt.xlabel("Epoch")
        plt.ylabel(aspect)

        plt.title("Training and Validation " + aspect)

        if aspect == "loss":
            plt.legend(loc = "upper right")
        else:
            plt.legend(loc = "lower right")
    
    if to_file:
        file_name = ""
        if not model_name is None:
            file_name += model_name
        else:
            # If there is no model name, add the time in the format HH:MM
            file_name += datetime.datetime.now().strftime("%H:%M")
        file_name += ".png"

        ensure_directory_exists(root_dir)

        plt.savefig(os.path.join(root_dir, file_name))
    else:
        plt.show()

def get_unet_model(output_channels: int, down_stack, up_stack):
    '''
    Returns a U-Net model.

    Parameters:
        output_channels (int): The number of output channels.
        down_stack (list): The down stack of the U-Net.
        up_stack (list): The up stack of the U-Net.

    Returns:
        A U-Net model.

    Notes:
        The down stack should be a model with the following outputs:
        ```
        [
            (batch_size, 64, 64, 64),
            (batch_size, 32, 32, 128),
            (batch_size, 16, 16, 256),
            (batch_size, 8, 8, 512),
            (batch_size, 4, 4, 512)
        ]
        ```
        The up stack should be a model with the following outputs:
        ```
        [
            (batch_size, 8, 8, 1024),
            (batch_size, 16, 16, 512),
            (batch_size, 32, 32, 256),
            (batch_size, 64, 64, 128)
        ]
        ```
    '''
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

class AugmentLayer(tf.keras.layers.Layer):
    '''
    Implements a layer that augments the inputs and masks.

    Methods:
        `__init__(self, seed = 42)` -> `None`
        `call(inputs, mask)` -> `(inputs, mask)`
            Augments the inputs and masks.

    Notes:
        The augmentation is a random horizontal flip.

    '''
    def __init__(self, seed = 42):
        super().__init__()

        # Both layers should have the same seed so they augment in tandem
        self.augment_inputs = tf.keras.layers.RandomFlip(mode = "horizontal", seed = seed)
        self.augment_mask = tf.keras.layers.RandomFlip(mode = "horizontal", seed = seed)

        self.augment_inputs = tf.keras.layers.RandomFlip(mode = "vertical", seed = seed)
        self.augment_mask = tf.keras.layers.RandomFlip(mode = "vertical", seed = seed)

    def call(self, inputs, mask):
        inputs = self.augment_inputs(inputs)
        mask = self.augment_mask(mask)

        return inputs, mask