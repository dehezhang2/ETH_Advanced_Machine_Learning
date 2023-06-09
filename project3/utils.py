import pickle
import gzip
import numpy as np
import os
import cv2

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def evaluate_frame(prediction, target):
    ious = []
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    for i in range(prediction.shape[0]):

        overlap = prediction[i]*target[i]
        union = prediction[i] + target[i]
        iou = overlap.sum()/float(union.sum())
        ious.append(iou)
    return np.median(ious)

def evaluate_video(predictions, targets):
    ious = []
    for p, t in zip(predictions, targets):
        assert p['name'] == t['name']
        prediction = np.array(p['prediction'], dtype=bool)
        target = np.array(t['label'], dtype=bool)

        assert targets.shape == predictions.shape
        overlap = predictions*targets
        union = predictions + targets

        ious.append(overlap.sum()/float(union.sum()))
    
    print("Median IOU: ", np.median(ious))
    return(np.median(ious))
    
def is_labeled(mask_list):
    
    label_index = []
    for i in range((mask_list).shape[2]):
        if(True in mask_list[:,:,i]):
            label_index.append(i)
            continue
    return np.array(label_index)

def image_size_normalize(raw_data, size):
    img_height, img_width, img_channel = size
    
    labeled_img_list = []
    mask_list = []
    
    for patient_i in range(len(raw_data)):

        img_i = raw_data[patient_i]['video'] # Shape: [IMG_HEIGHT1, IMG_WIDTH1, Frame_Num1]
        mask_i = raw_data[patient_i]['label']

        label_index = is_labeled(mask_i)
        labeled_img_i = img_i[:,:,label_index]
        labeled_mask_i = mask_i[:,:,label_index]
        
        for frame_j in range(labeled_mask_i.shape[2]):
            img_j = labeled_img_i[:,:,frame_j] # [IMG_HEIGHT2, IMG_WIDTH2, Frame_Num2]
            mask_j = labeled_mask_i[:,:,frame_j]
            mask_j_digi = np.zeros(mask_j.shape)
            mask_j_digi[mask_j] = 1
            if(img_j.shape[0]!= img_height or img_j.shape[1]!= img_width):
                img_j = cv2.resize(img_j,(img_height, img_width))
                mask_j_digi = cv2.resize(mask_j_digi,(img_height, img_width), interpolation = cv2.INTER_NEAREST)
            labeled_img_list.append(img_j)
            mask_list.append(mask_j_digi)
            
    return np.array(labeled_img_list).reshape(-1, img_height, img_width, img_channel), np.array(mask_list).reshape(-1,img_height, img_width, 1)
