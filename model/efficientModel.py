import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.utils import class_weight



image_resize = (120,120)
train_ds = keras.utils.image_dataset_from_directory(
    "dataset/foods-ss/train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_resize,
    batch_size=64
)
val_ds = keras.utils.image_dataset_from_directory(
    "dataset/foods-ss/train",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_resize,
    batch_size=64
)
test_ds = keras.utils.image_dataset_from_directory(
    "dataset/foods-ss/validation",
    seed=123,
    image_size=image_resize,
    batch_size=64
)

class_names = train_ds.class_names
print(class_names)

#I am going to pass in weights I personally calculated bassed off the following formula:
# (1/NumOfPicsInClass)*(totalPics/2)
train_class_weights = {0: 13.8, 1: 0.257, 2: 8.9, 3: 10.3, 4: 10.8, 5: 8.4, 6: 9.8, 7: 27.16, 8: 12.4, 9: 12.7, 10: 21.68}
print("Class WEights: ",train_class_weights)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal_and_vertical",
                      input_shape=(120,120,3)),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
  ]
)

preprocess_input = tf.keras.applications.efficientnet.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = image_resize + (3,)
base_model = tf.keras.applications.EfficientNetB1(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# Freeze model so we don't retrain it
base_model.trainable = False 

#turn image features into a simple vector
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

#convert to number of classes - ie computer will now guess with class an image belongs to
prediction_layer = tf.keras.layers.Dense(11)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(120, 120, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0004
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

len(model.trainable_variables)


initial_epochs = 8 #plato after 8 epochs

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds,
                    class_weight=train_class_weights)

#now we finetune:------

base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

fine_tune_epochs = 15 #platos after this
total_epochs =  initial_epochs + fine_tune_epochs

def lr_scheduler(epoch, lr):
  if epoch <initial_epochs+15:
    learning_rate = base_learning_rate/10
  elif epoch <initial_epochs+30:
    learning_rate = base_learning_rate/15
  else:
    learning_rate = base_learning_rate
  return learning_rate

lr_sched_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,verbose=1)

history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_ds,
                         callbacks= lr_sched_callback,
                         class_weight=train_class_weights)

model.evaluate(test_ds)

acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.5, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig("SelectiveSWeighted")

model.save('saved_model/SelectiveSWeighted.keras')
#finishes T98 and V93
print("Program End")