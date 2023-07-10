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
    print(example["metadata"])
    print(example["filepath"])

    plt.figure(figsize=(7, 7))
    for index, detection in enumerate(detections):
        mask = detection["mask"]
        print(len(mask), len(mask[0]))
        print(detection)
        plt.subplot(1, len(detections), index + 1)
        plt.imshow(mask)
    plt.savefig("segmentation_data/masks.png")

# Coco stores the bounds as [y0, x0, y1, x1] and as a ratio of the image size