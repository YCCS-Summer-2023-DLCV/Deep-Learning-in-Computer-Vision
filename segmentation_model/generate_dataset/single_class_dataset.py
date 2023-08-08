from generate_dataset import generate_dataset

classes = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]

single_class = ["banana"]

min_size = (30, 30)

generate_dataset(ds_name = "banana-segmentation-dataset", split = "train", classes = single_class, min_size = min_size)
generate_dataset(ds_name = "banana-segmentation-dataset", split = "validation", classes = single_class, min_size = min_size)
