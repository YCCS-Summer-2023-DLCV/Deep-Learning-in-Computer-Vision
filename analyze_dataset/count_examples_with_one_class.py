import fiftyone.zoo as foz
import fiftyone as fo
import os
from tqdm import tqdm

def _set_path():
    # Add the root of the venv to the path
    # This is necessary to import the selective_search module

    # Get the path to the venv
    venv_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Add the venv path to the path
    import sys
    sys.path.append(venv_path)

_set_path()

from segmentation_model.generate_dataset.example import Example


def count_examples_with_one_class(classes, split):
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split = split,
        classes = labels,
    )

    examples_with_one_class = 0

    for example in tqdm(dataset):
        if exactly_one_class(example, classes):
            examples_with_one_class += 1
    
    session = fo.Session()
    
    return examples_with_one_class, session

def exactly_one_class(example, classes):
    overlap = set()

    example = Example(example)

    for label in example.get_labels():
        if label in classes:
            overlap.add(label)
        
        if len(overlap) > 1:
            return False
    
    if len(overlap) == 0:
        return False
    return True
            



        

if __name__ == "__main__":
    labels = ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]
    
    val_count, _ = count_examples_with_one_class(labels, "train")
    print("There are", val_count, "examples in coco-2017 with only one of the food classes.")