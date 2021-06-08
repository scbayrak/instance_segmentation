import torch
import numpy as np

def get_box_sizes(boxes):
    # boxes in [x1, y1, x2, y2] format
    # height = y2 - y1, width = x2 - x1
    height, width = boxes[:,3] - boxes[:,1], boxes[:,2] - boxes[:,0]
    size = height * width
    return size


def box_iou(gt_box, pred_boxes, gt_box_size, pred_boxes_sizes):
    # calculate gt_box intersection y1, y2, x1, x2 positions for each pred box
    y1 = torch.max(gt_box[1], pred_boxes[:, 1])
    y2 = torch.min(gt_box[3], pred_boxes[:, 3])
    x1 = torch.max(gt_box[0], pred_boxes[:, 0])
    x2 = torch.min(gt_box[2], pred_boxes[:, 2])
    zeros = torch.zeros(pred_boxes.size()[0], device=device)
    # calculate intersection size for each prediction box
    intersections = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)
    union = gt_box_size+pred_boxes_sizes-intersections
    ious = intersections/union
    return ious


def compute_overlaps_masks(masks1, masks2):
    # masks1: (HxWxnum_pred)
    # masks2: (HxWxnum_gts)
    # flatten masks and compute their areas
    # masks1: num_pred x H*W
    # masks2: num_gt x H*W
    # overlap: num_pred x num_gt
    masks1 = masks1.flatten(start_dim=1)
    masks2 = masks2.flatten(start_dim=1)
    # sum the rows to find total area for each gt and prediction
    area2 = masks2.sum(dim=(1,), dtype=torch.float)
    area1 = masks1.sum(dim=(1,), dtype=torch.float)
    # duplicate each predicted mask num_gt times, compute the union (sum) of areas
    # num_pred x num_gt
    area1 = area1.unsqueeze_(1).expand(*[area1.size()[0], area2.size()[0]])
    union = area1 + area2
    # intersections and union: transpose predictions, the overlap matrix is num_predxnum_gts
    intersections = masks1.float().matmul(masks2.t().float())
    #print('inter', intersections, area1, area2)
    # +1: divide by 0
    overlaps = intersections / (union-intersections)
    return overlaps


def compute_matches(gt_boxes=None, gt_class_ids=None, gt_masks=None,
                    pred_boxes=None, pred_class_ids=None, pred_scores=None, pred_masks=None,
                    iou_threshold=0.5):

    # Compute IoUs [pred_masks, gt_masks]
    ious = compute_overlaps_masks(pred_masks, gt_masks)
    # separate predictions for each gt object
    split_ious = ious.t().split(1)
    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    # At the start all predictions are False Positives, all gts are False Negatives
    pred_match = (torch.ones(pred_boxes.size()[0])*-1).float()
    gt_match = (torch.ones(gt_boxes.size()[0])*-1).float()

    for _i, single_gt_ious in enumerate(split_ious):
        # ground truth class
        gt_class = gt_class_ids[_i]
        if (single_gt_ious>iou_threshold).any():
           # get best predictions, their indices in the IoU tensor and their classes
           global_best_preds_inds = torch.nonzero(
           single_gt_ious[0]>iou_threshold).view(-1)# tensor containing the indices of all non-zero elements of input
           pred_classes = pred_class_ids[global_best_preds_inds]
           best_preds = single_gt_ious[0][single_gt_ious[0]>iou_threshold]
           #  sort them locally-nothing else,
           local_best_preds_sorted = best_preds.argsort().flip(dims=(0,))
           # loop through each prediction's index, sorted in the descending order
           for p in local_best_preds_sorted:
               if pred_classes[p]==gt_class:
                  # Hit?
                  match_count +=1
                  pred_match[global_best_preds_inds[p]] = _i
                  gt_match[_i] = global_best_preds_inds[p]
                  # important: if the prediction is True Positive, finish the loop
                  break

    return gt_match, pred_match, ious


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):

    # Get matches and ious
    gt_match, pred_match, ious = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = (pred_match>-1).cumsum(dim=0).float().div(torch.arange(pred_match.numel()).float()+1)
    recalls = (pred_match>-1).cumsum(dim=0).float().div(gt_match.numel())
    # Pad with start and end values to simplify the math
    precisions = torch.cat([torch.tensor([0]).float(), precisions, torch.tensor([0]).float()])
    recalls = torch.cat([torch.tensor([0]).float(), recalls, torch.tensor([1]).float()])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
    # Compute mean AP over recall range
    indices = torch.nonzero(recalls[:-1] !=recalls[1:]).squeeze_(1)+1
    AP = torch.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    return AP, precisions, recalls, ious

def compute_ap_iou_range(gt_boxes, gt_class_ids, gt_masks,
            pred_boxes, pred_class_ids, pred_scores, pred_masks):

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_APs = []
    for threshold in iou_thresholds:

        AP, precisions, recalls, ious = compute_ap(
                gt_boxes, gt_class_ids, gt_masks, pred_boxes, 
                pred_class_ids, pred_scores, pred_masks, threshold)

        all_APs.append(AP)
    
    MAP = np.array(all_APs).mean()

    return MAP

def compute_recall(pred_masks, gt_masks, iou):

    ious = compute_overlaps_masks(pred_masks, gt_masks)
    # find max iou for each prediction box and its index
    max_vals, gt_classes = torch.max(ious, axis=1)[0], torch.max(ious, axis=1)[1]
    # filter the indexes where iou is bigger than threshold
    ids_above_thres = torch.where(max_vals >= iou)[0]
    # find the gt_boxes that have a match 
    matched_gt_boxes = gt_classes[ids_above_thres]
    # calculate the recall
    recall = list(torch.unique(matched_gt_boxes).size())[0] / gt_masks.shape[0]

    return recall

def compute_recall_iou_range(pred_masks, gt_masks):

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_Rs = []

    for threshold in iou_thresholds:
        recall = compute_recall(pred_masks, gt_masks, threshold)
        all_Rs.append(recall)
    
    MAR = np.array(all_Rs).mean()

    return MAR

