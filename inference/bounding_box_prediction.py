import numpy as np
import os
import tensorflow as tf
import selective_search
from PIL import Image
import cv2

labels = ["apple","background", "banana",  "broccoli" ,  "cake", "carrot", "donut", "hot dog", "orange", "pizza", "sandwich"]

def _predict_image(image_path, model_path):
 
    # Recreate the exact same model, including its weights and the optimizer
    img = Image.open(image_path)
    new_model = tf.keras.models.load_model(model_path)

    # Show the model architecture
    new_model.summary()
    image_bbox = selective_search.selective_search_fast(image_path)
    scores = dict()
    sorted_bbox = dict()
    for i in range(0,11):
        scores[i] = list()
        sorted_bbox[i] = list()
    for i in image_bbox:
        cropped_image = _crop_image(img,i)
        cropped_image  = tf.keras.utils.img_to_array(cropped_image)
        cropped_image = tf.image.resize(cropped_image,(120,120))
        cropped_image = tf.reshape(cropped_image, [1, 120, 120, 3])
        prediction = new_model.predict(cropped_image)
        prediction_max = np.max(prediction[0])
        test = np.argmax(prediction[0])
        scores[test].append(prediction_max)
        sorted_bbox[test] = np.append(sorted_bbox[test], (i[1],i[0],i[1]+i[3],i[0]+i[2]), axis=0)
       
    return sorted_bbox,scores

def _crop_image(example, bbox):
    bbox_rect = (bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])
    return example.crop(bbox_rect)

def _non_max_suppression(boxes, scores, max_bboxes, iou_threshold, score_threshold):
    nms_boxes = dict()    
    for i in range(0,11):
        length = len(scores[i])
        if i in boxes and not i == 1 :
            box = np.asarray(boxes[i], dtype=np.int32).reshape(length,4)
            score = np.asarray(scores[i])
            selected_indices = tf.image.non_max_suppression(box, score,max_bboxes,iou_threshold, score_threshold)
            nms_boxes[i] = tf.gather(box, selected_indices).numpy()
    return nms_boxes

def _draw_bbox(boxes, image_path, final_path):
    image = cv2.imread(image_path)
    for i in range (0,11):
        if i in boxes:
            for (startY, startX, endY, endX) in boxes[i]:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 1)
                (w, h), _ = cv2.getTextSize(
                    labels[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                image = cv2.rectangle(image, (startX, startY - 20), (startX + w, startY), (0, 0, 255), -1)
                image = cv2.putText(image, labels[i], (startX, startY - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
    cv2.imwrite(final_path, image)

def predict_image_with_nms(image_path,final_path,  model_path, max_bboxes, iou_threshold, score_threshold):
    sorted_bbox,scores = _predict_image(image_path,model_path)
    box = _non_max_suppression(sorted_bbox,scores,max_bboxes, iou_threshold,score_threshold)
    _draw_bbox(box,image_path,final_path)

def predict_image_without_nms(image_path,final_path,  model_path):
    sorted_bbox,scores  = _predict_image(image_path,model_path)
    boxes= dict()
    for i in range (0,11):
        if i in sorted_bbox and not i == 1:
            length = len(scores[i])
            boxes[i] = np.asarray(sorted_bbox[i], dtype=np.int32).reshape(length,4)
    _draw_bbox(sorted_bbox,image_path,final_path)

image_names = ["apple","sandwich","carrot","broccoli", "pizza"]

for name in image_names:
    for i in range(1,5):
        predict_image_with_nms("/home/ec2-user/Deep-Learning-in-Computer-Vision/inference/coco_test/"+name+"_"+ str(i)+".jpg", "/home/ec2-user/Deep-Learning-in-Computer-Vision/inference/coco_inference_10.5/"+name+"_"+ str(i)+"_final.jpg","/home/ec2-user/Deep-Learning-in-Computer-Vision/model/saved_model/SelectiveSWeighted.keras",100,0.3,10.5)