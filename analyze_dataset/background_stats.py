import tensorflow_datasets as tfds
from tqdm import tqdm

# count the amount of images without our labels
def count_images_without_labels(labels, dataset):
    count = 0
    for example in tqdm(dataset):
        labels_in_example = example["objects"]["label"].numpy()
        for label in labels_in_example:
            if label in labels:
                count += 1
                break
        
    return len(dataset) - count

# determine how many crops need to be taken from each background image

def _get_label_ids(user_labels, all_labels):
    id_subset = []

    for user_label in user_labels:
        for index, coco_label in enumerate(all_labels):
            if user_label == coco_label:
                id_subset.append(index)
    
    return id_subset

train_ds, info = tfds.load("coco/2017", split = "train", with_info = True)
all_labels = info.features["objects"]["label"].names
labels = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]
label_ids = _get_label_ids(labels, all_labels)

#print(count_images_without_labels(label_ids, train_ds))

val_ds = tfds.load("coco/2017", split = "validation")
print(len(val_ds))
print(count_images_without_labels(label_ids, val_ds))