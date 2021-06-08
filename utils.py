import skimage.io as IO
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import seaborn as sns

import dataset as Dataset

from pycocotools.coco import COCO
from cv2 import cv2
from PIL import Image

CATEGORIES =  [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']




def load_image(image_path):
    image = IO.imread(image_path)
    image = Dataset.CocoDataset.transform(image) 
    return image

def random_color():
    return np.random.choice(256, 3,replace=True)

def get_coloured_mask(mask):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    chosen_colour = list(map(int, random_color()))
    r[mask == 1], g[mask == 1], b[mask == 1] = chosen_colour
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask, chosen_colour

def segment_instance(pred, img_path, confidence=0.5, text_thickness=0.3):

    scores = pred[0]['scores'].detach().cpu().numpy()
    masks = pred[0]['masks'].squeeze().detach().cpu().numpy()
    labels = list(pred[0]['labels'].detach().cpu().numpy())
    boxes = list(pred[0]['boxes'].detach().cpu().numpy())

    # confident threshold index

    prediction_threshold = np.where(scores>confidence)[0].shape[0]
  
    # generate the all labels from index and then filter by using the threshold index.
    prediction_classes = [CATEGORIES[i] for i in labels][:prediction_threshold]
    
    # generate the all boxes from index and then filter by using the threshold index.
    prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in boxes][:prediction_threshold]
    
    # get the masks, convert it to a bool array to use it coloring. 
    prediction_masks = (masks>confidence)[:prediction_threshold]


    # display images
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(prediction_masks)):
        # box thickness proportionate to the confidence score
        #box_thickness=1
        box_thickness = math.ceil((scores[i]-confidence)/0.1)
        rgb_mask, colour = get_coloured_mask(prediction_masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, prediction_boxes[i][0], prediction_boxes[i][1],color=colour, thickness=box_thickness)
        cv2.putText(img, prediction_classes[i] + ':' + str(round(scores[i],2)), prediction_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), text_thickness , cv2.LINE_AA)
    plt.figure(figsize=(20,10))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def evolution_metric_plot(stats):
    sns.set(rc={'figure.figsize':(10,3)})
    ax = sns.lineplot(data=stats['Loss'])
    ax.set(title="Loss") # title barplot
    plt.show()
    ax = sns.lineplot(data=stats['mAP'])
    ax.set(title="mAP") # title 
    plt.show()
    ax = sns.lineplot(data=stats['mAR'])
    ax.set(title="mAR") # title 
    plt.show()