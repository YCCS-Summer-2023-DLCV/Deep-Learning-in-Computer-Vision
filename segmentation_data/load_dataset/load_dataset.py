import os
import tensorflow as tf

def load_dataset(path_to_ds, split):
    # Load the dataset as a bunch of file paths
    path_to_examples = os.path.join(path_to_ds, split, "*", "*.jpeg")

    files = tf.data.Dataset.list_files(path_to_examples)
    print("Loaded files")

    for example in files.take(10):
        print("Heres an example")
        print(example)

if __name__ == "__main__":
    path = "/home/ec2-user/Documents/datasets/segmentation-dataset"
    load_dataset(path, "train")