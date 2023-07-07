import numpy as np
import os
import tensorflow as tf
import selective_search
from PIL import Image
import cv2

def predict_image(image_path, model_path,max_bboxes):
 
 # Recreate the exact same model, including its weights and the optimizer
    img = Image.open(image_path)
    new_model = tf.keras.models.load_model(model_path)

    # Show the model architecture
    new_model.summary()
    image_bbox = selective_search.selective_search_fast(image_path)
    labels = []
    index =0 
    for i in image_bbox:
        cropped_image = _crop_image(img,i)
        cropped_image  = tf.keras.utils.img_to_array(cropped_image)
        cropped_image = tf.image.resize(cropped_image,(120,120))
        cropped_image = tf.reshape(cropped_image, [1, 120, 120, 3])
        prediction = new_model.predict(cropped_image)
        prediction_max = np.max(prediction[0])
      
        labels.append(prediction_max)
        image_bbox[index] = (i[1],i[0],i[1]+i[3],i[0]+i[2])
        index += 1
       
    return _non_max_suppersion(image_bbox, np.asarray(labels), max_bboxes)

def _crop_image(example, bbox):
    bbox_rect = (bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])
    example = example.crop(bbox_rect)
    example.save("geeks.jpg")
    return example

def _non_max_suppersion(boxes, scores, max_bboxes):
    selected_indices = tf.image.non_max_suppression(boxes,scores,max_bboxes,iou_threshold=0.5)
    return tf.gather(boxes, selected_indices)
