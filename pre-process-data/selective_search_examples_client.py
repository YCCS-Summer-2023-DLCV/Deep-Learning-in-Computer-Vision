import tensorflow_datasets as tfds

from generate_selective_search_examples import generate_selective_search_examples

ds_train, info = tfds.load("coco/2017", split = "train", with_info = True)
all_labels = info.features["objects"]["label"].names
ds_val = tfds.load("coco/2017", split = "validation")

labels = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]

generate_selective_search_examples(ds_train.take(20), labels, all_labels, "foods-ss", "train")
#generate_selective_search_examples(ds_val, labels, all_labels, "foods-ss", "validation")