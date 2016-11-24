import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import os, sys
import numpy as np

import cv2
from glob  import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from get_boundingbox import process_extract
from utils import fit_image_into_frame
from network import build_filter_network

def calc_velocity(movementum, mass):
    return movementum/mass

def test_valid_boundary(boundry, search_boundry):
    min_x_t, min_y_t, max_x_t, max_y_t = boundry
    min_x, min_y, max_x, max_y = search_boundry
    if (max_x_t <= max_x and max_y_t <= max_y) and (min_x_t >= min_x and min_y_t >= min_y):
        return True
    return False

def get_surrouding_move(current_bb, search_boundry):
    MOVEMENTUM = 500
    min_x, min_y, max_x, max_y = current_bb

    velocity_x = calc_velocity(MOVEMENTUM, max_x-min_x)
    velocity_y = calc_velocity(MOVEMENTUM, max_y-min_y)
    
    result = []

    # right
    right_boundry = (min_x+velocity_x, min_y, max_x+velocity_x, max_y)
    if test_valid_boundary(right_boundry, search_boundry):
        # min_x_t1, min_y_t1, max_x_t1, max_y_t1 = right_boundry
        # print right_boundry, search_boundry
        # cv2.rectangle(sig_area, (min_x_t1, min_y_t1), (max_x_t1, max_y_t1), (255, 0, 0))
        result.append(right_boundry)

    # up
    up_boundry = (min_x, min_y-velocity_y, max_x, max_y-velocity_y)
    if test_valid_boundary(up_boundry, search_boundry):
        result.append(up_boundry)

    # down
    down_boundry = (min_x, min_y+velocity_y, max_x, max_y+velocity_y)
    if test_valid_boundary(down_boundry, search_boundry):
        result.append(down_boundry)

    # left
    left_boundry = (min_x-velocity_x, min_y, max_x-velocity_x, max_y)
    if test_valid_boundary(left_boundry, search_boundry):
        result.append(left_boundry)

    # right-up
    right_up_boundry = (min_x+velocity_x, min_y-velocity_y, max_x+velocity_x, max_y-velocity_y)
    if test_valid_boundary(right_up_boundry, search_boundry):
        result.append(right_up_boundry)

    # right-down
    right_down_boundry = (min_x+velocity_x, min_y+velocity_y, max_x+velocity_x, max_y+velocity_y)
    if test_valid_boundary(right_down_boundry, search_boundry):
        result.append(right_down_boundry)

    # left-up
    left_up_boundry = (min_x-velocity_x, min_y-velocity_y, max_x-velocity_x, max_y-velocity_y)
    if test_valid_boundary(left_up_boundry, search_boundry):
        result.append(left_up_boundry)

    # left-down
    left_down_boundry = (min_x-velocity_x, min_y+velocity_y, max_x-velocity_x, max_y+velocity_y)
    if test_valid_boundary(left_down_boundry, search_boundry):
        result.append(left_down_boundry)


    return result

if __name__ == '__main__':
    filepaths = glob('data/FrontBW/*')

    batch_size = 32
    temp = 0

    # create placeholder for variables
    input_t = tf.placeholder(tf.float32, [None, 128, 128, 1])
    labels_t = tf.placeholder(tf.int64, shape=[None, ])
    dropout_prob_t = tf.placeholder(tf.float32)

    predict, _ = build_filter_network(input_t, labels_t, dropout_prob_t)

    print "Loading model ..."
    saver = tf.train.Saver()
    sess = tf.Session()
    tf.initialize_all_variables()
    saver.restore(sess, '/home/arya_01/AxisProject/AxisFilter/model/00213_FILTER/model')
    print "Sucessfuly loaded model ..."
    with sess.as_default():
        for _id, fp in enumerate(filepaths):
            img = cv2.imread(fp, 0)
            img = fit_image_into_frame(img, frame_size=(736, 1600, 1), random_fill=False, fill_color=255, mode='fit')
            sig_area  = img[350:610, 800:1600]
            
            select = []
            for bb_batch in process_extract(sig_area, batch_size=batch_size):
                new_bb_batch = []
                for bb in bb_batch:
                    min_x, min_y, max_x, max_y = bb
                    extract_data = sig_area[min_y:max_y, min_x: max_x]
                    extract_data = fit_image_into_frame(extract_data, frame_size=(128, 128, 1), random_fill=False, fill_color=255, mode='fit')
                    new_bb_batch.append(extract_data)
                predict_data = predict.eval(feed_dict={
                input_t: np.asarray(new_bb_batch),
                dropout_prob_t: 1.0})
                for pb, p, bb in zip(predict_data, np.argmax(predict_data, axis=1), bb_batch):
                    if p == 1:
                        select.append(bb)

            search_boundry = (0, 0, sig_area.shape[1], sig_area.shape[0])
            sig_area = cv2.cvtColor(sig_area, cv2.COLOR_GRAY2BGR)
            for bb in select:
                bb_new = get_surrouding_move(bb, search_boundry)
                if bb_new == []:
                    min_x, min_y, max_x, max_y = bb
                    cv2.rectangle(sig_area, (min_x, min_y), (max_x, max_y), (0, 0, 255))
                else:
                    for bb_n in bb_new:
                        min_x, min_y, max_x, max_y = bb_n
                        cv2.rectangle(sig_area, (min_x, min_y), (max_x, max_y), (255, 0, 0))
                    min_x, min_y, max_x, max_y = bb
                    cv2.rectangle(sig_area, (min_x, min_y), (max_x, max_y), (0, 255, 0))
            cv2.imwrite('tmp/imgs/'+str(temp)+'.png', sig_area)
            temp += 1

