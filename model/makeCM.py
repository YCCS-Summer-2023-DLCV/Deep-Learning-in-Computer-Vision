import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
# Assuming you have a pre-existing model called 'model'
# Load the dataset
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
train_dir = "dataset/train"
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
model = keras.models.load_model('/home/ec2-user/ralph/project/saved_model/Efficient2mk2/saved_model.pb')
# Retrieve the true labels from the dataset
true_labels = []
for images, labels in train_ds:
    true_labels.extend(labels.numpy())
true_labels = np.array(true_labels)
# Generate predictions using the model
predictions = model.predict(train_ds)
predicted_labels = np.argmax(predictions, axis=1)
# Create the confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)
# Display the confusion matrix
class_names = train_ds.class_names
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion_mat, cmap='Blues')
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
# Rotate the x-axis labels for better visibility
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# Loop over data dimensions and create text annotations
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, confusion_mat[i, j], ha="center", va="center", color="white")
ax.set_title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("ef2Matrix.jpeg")