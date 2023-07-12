import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.utils.multiclass import unique_labels
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
# Assuming you have a pre-existing model called 'model'
model = tf.keras.models.load_model("saved_model/efficientnet_b1")

# Load the dataset
BATCH_SIZE = 32
IMG_SIZE = (120, 120)
test_dir = "validation"

test_generator = ImageDataGenerator()
test_data_generator = test_generator.flow_from_directory(
    test_dir, # Put your path here
     target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False)
test_steps_per_epoch = numpy.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
# Get most likely class
predicted_classes = numpy.argmax(predictions, axis=1)

true_classes = test_data_generator.classes
class_labels = list(test_data_generator.class_indices.keys())   

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report) 

cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_labels, fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(class_labels, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Refined Confusion Matrix', fontsize=20)

plt.savefig('ConMat24.png')