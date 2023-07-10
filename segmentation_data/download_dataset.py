import fiftyone.zoo as foz
import matplotlib.pyplot as plt

labels = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split = "validation",
    max_samples = 1,
    classes = labels,
    label_types = ["segmentations"]
)

for example in dataset.take(1):
    detections = example["ground_truth"]["detections"]
    print(example)

    plt.figure(figsize=(7, 7))
    for index, detection in enumerate(detections):
        mask = detection["mask"]
        plt.subplot(1, len(detections), index + 1)
        plt.imshow(mask)
    plt.savefig("segmentation_data/masks.png")