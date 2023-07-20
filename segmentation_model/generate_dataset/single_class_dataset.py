from generate_dataset import generate_dataset

classes = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]

single_class = ["broccoli"]

generate_dataset(ds_name = "brocolli-segmentation-dataset", split = "train", classes = single_class)
generate_dataset(ds_name = "brocolli-segmentation-dataset", split = "validation", classes = single_class)