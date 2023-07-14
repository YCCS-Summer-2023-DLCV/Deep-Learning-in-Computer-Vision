import os
import tensorflow as tf

IMG_SIZE = [128, 128]

def load_dataset(path_to_ds, split):
    # Load the dataset as a bunch of file paths
    path_to_examples = os.path.join(path_to_ds, split, "*.jpeg")

    files_for_ds = tf.data.Dataset.list_files(path_to_examples)
    print("Loaded files")
    
    dataset = files_for_ds.map(process_path, num_parallel_calls = tf.data.AUTOTUNE)

    return dataset

def get_path_to_mask(path_to_image):

    # The paths are in this format:
    # /home/ec2-user/Documents/datasets/segmentation-dataset/train/643.jpeg
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
    path_to_mask = get_path_to_mask(file_path)

    image = tf.io.read_file(file_path)
    mask = tf.io.read_file(path_to_mask)

    image = decode_image(image)
    mask = decode_image(mask)

    return image, mask

if __name__ == "__main__":
    path = "/home/ec2-user/Documents/datasets/segmentation-dataset"
    dataset = load_dataset(path, "train")

    for example in dataset.take(1):
        print(example)