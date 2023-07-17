import os
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = [128, 128]

def load_dataset(path_to_ds, split):
    # Load the dataset as a bunch of file paths
    path_to_images = os.path.join(path_to_ds, split, "*.jpeg")
    path_to_masks = os.path.join(path_to_ds, split + "-anno", "*.jpeg")

    images = tf.data.Dataset.list_files(path_to_images, shuffle = False)
    masks = tf.data.Dataset.list_files(path_to_masks, shuffle = False)

    images = images.map(process_path, num_parallel_calls = tf.data.AUTOTUNE)
    masks = masks.map(process_path, num_parallel_calls = tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.zip((images, masks))

    return dataset

def get_path_to_mask(path_to_image):

    # The paths are in this format:
    # /home/ec2-user/Documents/datasets/segmentation-dataset/train/64b195d0249c1827cddd4fb3.jpeg
    # It needs to turn into:
    # /home/ec2-user/Documents/datasets/segmentation-dataset/train-anno/64b195d0249c1827cddd4fb3.jpeg
    # We need to get the directory and append "-anno"

    print(path_to_image)

    directory = os.path.dirname(path_to_image)
    file_name = os.path.basename(path_to_image)

    directory += "-anno"

    path_to_mask = os.path.join(directory, file_name)

    return path_to_mask

def decode_image(image):
    image = tf.io.decode_jpeg(image)

    return tf.image.resize(image, IMG_SIZE)

def process_path(file_path):
    image = tf.io.read_file(file_path)

    image = decode_image(image)

    return image

if __name__ == "__main__":
    path = "/home/ec2-user/Documents/datasets/segmentation-dataset"
    dataset = load_dataset(path, "train")

    for example in dataset.take(1):
        print(example)

        image, mask = example

        plt.figure(figsize = (7, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(image.numpy().astype("uint8"))

        plt.subplot(1, 2, 2)
        plt.imshow(mask.numpy())

        plt.savefig("plot.png")