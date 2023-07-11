import fiftyone.zoo as foz
import matplotlib.pyplot as plt

from process_example import process_example

labels = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split = "validation",
    max_samples = 1,
    classes = labels,
    label_types = ["segmentations"],
    shuffle = True,
    seed = 52
)

for example in dataset.take(1):
    detections = example["ground_truth"]["detections"]
    print(example["metadata"])
    print(example["filepath"])

    # plt.figure(figsize=(7, 7))
    # for index, detection in enumerate(detections):
    #     mask = detection["mask"]
    #     plt.subplot(1, len(detections), index + 1)
    #     plt.imshow(mask)


    plt.savefig("segmentation_data/masks.png")

    processed_examples = process_example(example, labels)
    plt.figure(figsize=(7, 7))
    for index, processed_example in enumerate(processed_examples):
        plt.title(processed_example["label"])
        ax = plt.subplot(len(processed_examples), 2, (index * 2) + 1)
        plt.imshow(processed_example["image"])
        plt.axis("off")

        ax.remove()

        plt.title(processed_example["label"])
        plt.subplot(len(processed_examples), 2, (index * 2) + 2)
        plt.imshow(processed_example["mask"])
        plt.axis("off")

    plt.savefig("segmentation_data/plots/processed_example.png")

# Coco stores the bounds as [y0, x0, y1, x1] and as a ratio of the image size