import numpy as np
import os
import tensorflow as tf
import selective_search
from PIL import Image
import cv2
import tensorflow_addons as tfa

labels = ["apple","background", "banana",  "broccoli" ,  "cake", "carrot", "donut", "hot dog", "orange", "pizza", "sandwich"]
labels_segmentation = ["broccoli"]
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

def _draw_bbox_with_segmentation(boxes, image_path, final_path, segmentation_model_path):
    image = cv2.imread(image_path)
    output = image.copy()

    for i in range (0,11):
        if i in boxes:
            for (startY, startX, endY, endX) in boxes[i]:
                if labels[i] in labels_segmentation:
                    _segmentaion(image_path,output,segmentation_model_path,startY,startX,endY,endX)     
                cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 1)
                (w, h), _ = cv2.getTextSize(
                    labels[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                output = cv2.rectangle(output, (startX, startY), (startX + w, startY +20), (0, 0, 255), -1)
                output = cv2.putText(output, labels[i], (startX, startY + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)                   
                 
    cv2.imwrite(final_path, output)


def _draw_bbox(boxes, image_path, final_path):
    image = cv2.imread(image_path)
    output = image.copy()

    for i in range (0,11):
        if i in boxes:
            for (startY, startX, endY, endX) in boxes[i]: 
                cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 1)
                (w, h), _ = cv2.getTextSize(
                    labels[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                output = cv2.rectangle(output, (startX, startY), (startX + w, startY +20), (0, 0, 255), -1)
                output = cv2.putText(output, labels[i], (startX, startY + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)                   
                 
    cv2.imwrite(final_path, output)
def _create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def _segmentaion(image_path,output,model_path,startY,startX,endY,endX):
    image = Image.open(image_path)
    bbox_rect = (startX,startY,endX,endY)
    cropped_image = image.crop(bbox_rect)
    model = tf.keras.models.load_model(model_path)
    cropped_image  = tf.keras.utils.img_to_array(cropped_image)
    cropped_image = tf.image.resize(cropped_image,(128,128))
    cropped_image_resize = tf.reshape(cropped_image, [1, 128, 128, 3])
    mask_tensorflow = _create_mask(model.predict(cropped_image_resize))
    mask = mask_tensorflow.numpy().astype("uint8")
    mask = cv2.resize(mask, (endX-startX, endY-startY), interpolation=cv2.INTER_NEAREST)
    mask[mask==0] = 2
    mask[mask==1] = 0
    mask[mask==2] = 1

    color = np.array([255, 0, 0], dtype='uint8')
    roi = output[startY:endY, startX:endX]
    # randomly select a color that will be used to visualize this
    # particular instance segmentation then create a transparent
    # overlay by blending the randomly selected color with the ROI
    stacked_mask = np.stack((mask,)*3, axis=-1)
    masked_img  = np.where(stacked_mask, roi, color)
    blended = cv2.addWeighted(roi, 0.8, masked_img, 0.2,0)

    output[startY:endY, startX:endX] = blended

def predict_image_with_nms_with_segmentation(image_path,final_path,  model_path, max_bboxes, iou_threshold, score_threshold, segmentation_model_path):
    sorted_bbox,scores = _predict_image(image_path,model_path)
    box = _non_max_suppression(sorted_bbox,scores,max_bboxes, iou_threshold,score_threshold)
    _draw_bbox_with_segmentation(box,image_path,final_path, segmentation_model_path)


def predict_image_with_nms(image_path,final_path,  model_path, max_bboxes, iou_threshold, score_threshold, segmentation_model_path):
    sorted_bbox,scores = _predict_image(image_path,model_path)
    box = _non_max_suppression(sorted_bbox,scores,max_bboxes, iou_threshold,score_threshold)
    _draw_bbox(box,image_path,final_path)

def predict_image_without_nms(image_path,final_path,  model_path):
    sorted_bbox,scores  = _predict_image(image_path,model_path)
    boxes= dict()
    for i in range (0,11):
        if i in sorted_bbox and not i == 1 and not i==0:
            length = len(scores[i])
            boxes[i] = np.asarray(sorted_bbox[i], dtype=np.int32).reshape(length,4)
    _draw_bbox(boxes,image_path,final_path)


#predict_image_with_nms("/home/ec2-user/Deep-Learning-in-Computer-Vision/inference/coco_test/carrots_and_broccoli_1.jpg", "/home/ec2-user/Deep-Learning-in-Computer-Vision/inference/carrot_broccoli_30_4-07/carrots_and_broccoli_1" +"_final.jpg","/home/ec2-user/Deep-Learning-in-Computer-Vision/model/saved_model/efficientnet_b1_focal_loss_no_overlap",100,0.30,4.07, "/home/ec2-user/Deep-Learning-in-Computer-Vision/segmentation_model/train_model/model.keras")

image_names = ["oranges_and_apples", "bananas", "carrots_and_broccoli"]


for name in image_names:
    for i in range(1,4):
        predict_image_without_nms("/home/ec2-user/Deep-Learning-in-Computer-Vision/inference/coco_test/"+name+"_"+ str(i)+".jpg", "/home/ec2-user/Deep-Learning-in-Computer-Vision/inference/final/without_nms_"+name+"_"+ str(i)+"_final.jpg","/home/ec2-user/Deep-Learning-in-Computer-Vision/model/saved_model/efficientnet_b1_focal_loss_no_overlap")
        predict_image_with_nms("/home/ec2-user/Deep-Learning-in-Computer-Vision/inference/coco_test/"+name+"_"+ str(i)+".jpg", "/home/ec2-user/Deep-Learning-in-Computer-Vision/inference/final/"+name+"_"+ str(i)+"_final.jpg","/home/ec2-user/Deep-Learning-in-Computer-Vision/model/saved_model/efficientnet_b1_focal_loss_no_overlap",100,0.25,3.55, "/home/ec2-user/Deep-Learning-in-Computer-Vision/segmentation_model/train_model/model.keras")