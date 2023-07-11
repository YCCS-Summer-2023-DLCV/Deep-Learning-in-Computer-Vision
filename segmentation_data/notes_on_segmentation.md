BS"D
# Segmentation with Coco

## Downloading the Dataset
Tensorflow does not have Coco's segmentation masks. Therefore, we need to download it with another software such as FiftyOne.

```python
labels = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split = "validation",
    max_samples = 1,
    classes = labels,
    label_types = ["segmentations"]
)
```

## Accessing the Masks
To access an example's mask, use `detections = example["ground_truth"]["detections"]`. This returns the data about each object in the image. This data is stored in a list - one element for each object. Each element's mask can be accessed with `element["mask"]`. This returns a boolean array.

```python
for example in dataset.take(1):
    detections = example["ground_truth"]["detections"]

    plt.figure(figsize=(7, 7))
    for index, detection in enumerate(detections):
        mask = detection["mask"]
        plt.subplot(1, len(detections), index + 1)

        plt.imshow(mask)

    plt.savefig("segmentation_data/masks.png")
```

The masks are sub-matricies of the original matrix. They correspond to the portion of the image which is in the bounding box.

The dimensions of the bounding boxes as stored by FiftyOne are in the following format: `[x, y, width, height]`.

## Plan for Creating Dataset
1. Use selective search to get boxes around each object. Make sure each box encompasses the coco bounding box.
2. Crop the image using the box.
3. Expand the mask to fill the whole box.
4. Save the image and the mask.

The path for each example will be `root_dir/split/class/<image id>-mask.png`