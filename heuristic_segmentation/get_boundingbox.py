import os, sys
import cv2
import numpy as np
from utils import *
import segment
import math

def filter(img):
    size_y, size_x = img.shape[:2]
    if size_x >= 10 and size_y >= 10:
        if size_x <= (260-26) and size_y <= (800-80):
            return True
    return False

def nms(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(x2)

    pick = []
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]
 
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
 
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
 
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
 
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
 
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
 
    # return only the bounding boxes that were picked
    return boxes[pick]

def process_extract(sig_area, batch_size=32):

    sig_area = cv2.cvtColor(sig_area, cv2.COLOR_GRAY2BGR)

    new_region_proposed_bb = []
    region_proposed_bb = segment.get_bounded_box(sig_area, 1.0, 320, 0)
    for bb in region_proposed_bb:
        min_x, min_y, max_x, max_y  = bb
        extract_data = sig_area[min_y:max_y, min_x: max_x]
        if filter(extract_data):
            new_region_proposed_bb.append(bb)

    new_region_proposed_bb = np.asarray(new_region_proposed_bb)
    pick = nms(new_region_proposed_bb, 0.1)

    num_batch = int(math.ceil(len(pick)/float(batch_size)))
    for batch_id in xrange(num_batch):
        yield(pick[batch_id * batch_size:(batch_id+1) * batch_size])

