import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def get_predictions(data, y_pred, threshold):
    # get the masks above threshold from the output
    masks = (y_pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # get the indexes above score threshold
    scores = y_pred[0]['scores'].detach().cpu().numpy()
    score_ind_above_thresh = np.where(scores>0.5)[0]
    score_no_above_threshold = score_ind_above_thresh.shape[0]
    # get only masks for objects with scores above threshold
    masks = masks[:score_no_above_threshold]
    # get a list of the boxes for objects with scores above threshold
    boxes = [[(box[0], box[1]), (box[2], box[3])] for box in y_pred[0]['boxes'].detach().cpu().numpy()]
    boxes = boxes[:score_no_above_threshold]
    # get the classes labels   
    labels = [data.all_categ_names[data.adjusted_category_ids[category_id.item()]] for category_id in y_pred[0]['labels']]
    return masks, boxes, labels

def apply_segmentation(image, data, masks, boxes, labels):
    mask_transparency = 0.6
    # create random colours for all categories
    colours= np.random.uniform(0, 255, size=(len(data.all_categ_names), 3))

    for mask in range(len(masks)):
        red = np.zeros_like(masks[mask]).astype(np.uint8)
        green = np.zeros_like(masks[mask]).astype(np.uint8)
        blue = np.zeros_like(masks[mask]).astype(np.uint8)
        # choose a colour randomly
        colour = colours[np.random.choice(len(colours))]
        # apply the colour to each RGB filter in mask locations
        red[masks[mask] == 1], green[masks[mask] == 1], blue[masks[mask] == 1]  = colour
        # combine the RGB filters
        segmentation = np.stack([red, green, blue], axis=2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, 1, segmentation, mask_transparency, 0)
        # draw the boxes 
        cv2.rectangle(image, boxes[mask][0], boxes[mask][1], color=colour, 
                      thickness=2)
        # add the label names
        cv2.putText(image , labels[mask]["name"], (int(boxes[mask][0][0]), int(boxes[mask][0][1]-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 
                    thickness=2, lineType=cv2.LINE_AA)

    plt.figure(figsize=(20,10))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return image