# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')
import cv2
import caffe_network
import numpy as np
import sys
from utils import *
from PIL import Image
import math
import matplotlib.pyplot as plt
import Queue
import multiprocessing
from multiprocessing.pool import ThreadPool


def run(augs):
    masked_img, img, patch_height, patch_width, index_y, index_x, y, x = augs
    img_rect = img.copy()
    cv2.rectangle(img_rect, (int(y*patch_height), int(x*patch_width)), (int((y+1)*patch_height), int((x+1)*patch_width)), (0, 0, 0), -1)
    masked_img.put((img_rect, index_y, index_x))

def get_heat_map(img, net, batch_data, patch_height, patch_width, overlap_ratio, label, batch_size=16):
    param_t = []
    masked_img = Queue.Queue()

    size_y = 0
    for index_y, y in enumerate(np.arange(0, math.ceil(img.shape[0]/float(patch_height)), overlap_ratio)):
        size_y += 1
        size_x = 0
        for index_x, x in enumerate(np.arange(0, math.ceil(img.shape[1]/float(patch_width)), overlap_ratio)):
            size_x += 1
            param_t.append((masked_img, img, patch_height, patch_width, index_y, index_x, y, x))

    pool = ThreadPool(1)
    pool.map(run, param_t)
    pool.close()
    pool.join()

    print masked_img.qsize(), 'number of images'

    heat_map = np.zeros((size_y, size_x))
    batch_matrix_index = []
    batch_data = []
    while not masked_img.empty():
        img, y, x = masked_img.get()

        batch_matrix_index.append((y, x))
        batch_data.append(img)

        if len(batch_data) >= batch_size:
            _, output = caffe_network.get_data(net, batch_data)
            output = output.copy()
            for prob, (y, x) in zip(output, batch_matrix_index):
                heat_map[y][x] = 1.-prob[label]
            batch_data = []
            batch_matrix_index = []


    if len(batch_data) > 0:
        _, output = caffe_network.get_data(net, batch_data)
        for prob, (y, x) in zip(output, batch_matrix_index):
            heat_map[y][x] = 1.-prob[label]
        batch_data = []
        batch_matrix_index = []

    # scale heate map from 0 to 1 range
    heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map)-np.min(heat_map))

    heat_map = np.asarray(heat_map*255, dtype='uint8')
    heat_map = cv2.resize(heat_map, (224, 224))
    return heat_map

if  __name__ == '__main__':
    model = "/home/arya_01/heat_map_feature/model/deploy.prototxt"
    weights = "/home/arya_01/heat_map_feature/model/bvlc_googlenet.caffemodel"
    net = caffe_network.create_network(model, weights)


    img = Image.open(sys.argv[1])

    img = np.asarray(img, dtype='uint8')
    img = create_fixed_image_shape(img, frame_size=(224, 224, 3))

    # predict label for image
    batch_data = [img]
    _, output = caffe_network.get_data(net, batch_data)
    output = output.copy()
    label = np.argmax(output)

    labels_list = []
    fr = open('model/labels.txt')
    for l in fr:
        l = l.strip()
        labels_list.append(l)

    print 'predicted', label, 'with prob ->', output[0][label], 'its label is ->', labels_list[label]




    overlap_ratio = 0.25

    # 10% width and 10% height img patch
    patch_height = int(round(img.shape[0] * 25/100.))
    patch_width = int(round(img.shape[1] * 25/100.))

    # scale image from 0 to 1 range
    heat_map = get_heat_map(img, net, batch_data, patch_height, patch_width, overlap_ratio, label)
    heat_map_img = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

    blend_img = cv2.addWeighted(img, 0.7, heat_map_img, 0.3, 0)

    cv2.imwrite('result_heatmap.png', blend_img)
