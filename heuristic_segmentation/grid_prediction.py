import matplotlib as mpl
mpl.use('Agg')

import os, sys
import numpy as np
import cv2
import math
from glob  import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from get_boundingbox import process_extract
from utils import fit_image_into_frame
from network import build_filter_network

def sliding_window(img_shape, stepSize, windowSize, batch_size=32):
    windows = []
    # slide a window across the image
    for y in xrange(0, img_shape[0], stepSize):
        for x in xrange(0, img_shape[1], stepSize):
            # yield the current window
            windows.append((x, y, x+windowSize[0], y+windowSize[1]))

    num_batchs  = int(math.ceil(len(windows)/float(batch_size)))
    for batch_id in xrange(num_batchs):
        yield windows[batch_id*batch_size: (batch_id+1)*batch_size]

def load_model(path):
    # create placeholder for variables
    input_t = tf.placeholder(tf.float32, [None, 128, 128, 1])
    labels_t = tf.placeholder(tf.int64, shape=[None, ])
    dropout_prob_t = tf.placeholder(tf.float32)

    predict, _ = build_filter_network(input_t, labels_t, dropout_prob_t)

    saver = tf.train.Saver()
    sess = tf.Session()
    tf.initialize_all_variables()
    saver.restore(sess, path)
    return locals()

if __name__ == '__main__':
    filepaths = glob('data/FrontBW/*')
    batch_size = 32

    # DEL_CODE
    temp = 0

    model_path = '/home/arya_01/AxisProject/AxisFilter/model/00239_FILTER/model'
    # load required model with all vairables
    model = load_model(model_path)

    input_t, dropout_prob_t, sess, predict = model['input_t'], model['dropout_prob_t'], model['sess'], model['predict']

    with sess.as_default():
        for fp in filepaths:
            img = cv2.imread(fp, 0)
            img = fit_image_into_frame(img, frame_size=(736, 1600, 1), random_fill=False, fill_color=255, mode='fit')
            sig_area  = img[350:350+260, 800:1600]
            for bb_batch in process_extract(sig_area):


                # predict 
                imgs = []
                for bb in bb_batch:
                    min_x, min_y, max_x, max_y = bb
                    # extract img
                    extract_data = sig_area[min_y:max_y, min_x: max_x]
                    extract_data = fit_image_into_frame(extract_data, frame_size=(128, 128, 1), random_fill=False, fill_color=255, mode='fit')
                    imgs.append(extract_data)

                predict_data = predict.eval(feed_dict={
                    input_t: np.asarray(imgs),
                    dropout_prob_t: 1.0})

                select = []
                for bb, p in zip(bb_batch, np.argmax(predict_data, axis=1)):
                    if p == 1:
                        select.append(bb)


                for bb in select:
                    min_x, min_y, max_x, max_y = bb
                    # extract img
                    extract_data = sig_area[min_y:max_y, min_x: max_x]
                    for windows_batch in sliding_window(extract_data.shape[:2], 64, (128, 128), batch_size):

                        imgs = []
                        for window in windows_batch:
                            min_x_w, min_y_w, max_x_w, max_y_w = window
                            img = extract_data[min_y_w:max_y_w, min_x_w: max_x_w]
                            img = fit_image_into_frame(img, frame_size=(128, 128, 1), random_fill=False, fill_color=255, mode='fit')
                            imgs.append(img)

                        predict_data = predict.eval(feed_dict={
                        input_t: np.asarray(imgs),
                        dropout_prob_t: 1.0})

                        extract_data = cv2.cvtColor(extract_data, cv2.COLOR_GRAY2BGR)
                        for pp, window in zip(np.argmax(predict_data, axis=1), windows_batch):
                            min_x_w, min_y_w, max_x_w, max_y_w = window
                            if pp == 1:
                                cv2.rectangle(extract_data, (min_x_w, min_y_w), (max_x_w, max_y_w), color=(0, 255, 0))
                            else:
                                cv2.rectangle(extract_data, (min_x_w, min_y_w), (max_x_w, max_y_w), color=(0, 0, 255))

                        cv2.imwrite('tmp/imgs/'+str(temp)+'.png', extract_data)
                        temp += 1