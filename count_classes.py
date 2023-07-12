import os

def count_classes(dataset_path):
    classes = os.listdir(dataset_path)
    class_counts = {}
    total_images = 0

    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            images = os.listdir(class_dir)
            num_images = len(images)
            class_counts[class_name] = num_images
            total_images += num_images

    # Display the class counts
    print("Class Counts:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    print(f"Total Images: {total_images}")

# Example usage
dataset_path = "train"  # Replace with the actual path to your dataset directory

count_classes(dataset_path)
