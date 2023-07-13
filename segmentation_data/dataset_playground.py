import fiftyone.zoo as foz
import fiftyone as fo
import matplotlib.pyplot as plt

from process_example import process_example

labels = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]

amt_of_images = 100

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split = "validation",
    max_samples = amt_of_images,
    classes = labels,
    label_types = ["segmentations"],
    shuffle = True,
    seed = 14
)

def predict_time_to_make_dataset(dataset, eventual_amt_of_examples):
    import time
    from tqdm import tqdm

    amt_in_ds = len(dataset)

    start_time = time.perf_counter()

    for example in dataset:
        process_example(example, labels, False)
    
    end_time = time.perf_counter()

    duration = end_time - start_time
    duration_per_example = duration / amt_in_ds

    predicted_time = duration_per_example * eventual_amt_of_examples

    return predicted_time

# train_predict_time = predict_time_to_make_dataset(dataset, 16255)
# print(f"Predicted time to process 16255 examples: {train_predict_time}")
# quit()

for example in dataset.take(1):
    detections = example["ground_truth"]["detections"]
    print(example["filepath"])
    print(example)
    quit()

    plt.figure(figsize=(7, 7))
    for index, detection in enumerate(detections):
        if not detection["label"] in labels:
            continue

        mask = detection["mask"]
        plt.subplot(len(detections), 1, index + 1)
        plt.imshow(mask)


    plt.savefig("segmentation_data/masks.png")

    processed_examples = process_example(example, labels, False)
    plt.figure(figsize=(7, 7))
    for index, processed_example in enumerate(processed_examples):
        ax = plt.subplot(len(processed_examples), 2, (index * 2) + 1)
        plt.imshow(processed_example["image"])
        plt.axis("off")

        ax = plt.subplot(len(processed_examples), 2, (index * 2) + 2)
        plt.imshow(processed_example["mask"])
        plt.axis("off")

    plt.savefig("segmentation_data/plots/processed_example.png")

# Coco stores the bounds as [y0, x0, y1, x1] and as a ratio of the image size
