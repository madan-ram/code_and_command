# RUN - python main.py [path to cheque data dir] [path to finicale data dir] [triplet model path] [binary model path]
# RUN SAMPLE -  python train_bin.py /home/aipocuser/Axis_Project/code/signature_training/output/cheque /home/aipocuser/Axis_Project/code/signature_training/output/finicale ./model
import os, sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from load_data import BinaryGenerator
from network import build_triplet_network, build_binary_classifier_network
from layers.core import input_data
import tensorflow as tf
from utils import fit_image_into_frame, test_create_dir
from predict import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

if __name__ == '__main__':
    cheque_path = sys.argv[1]
    finicale_path = sys.argv[2]
    model_triplet_path = sys.argv[3]
    model_binary_path = sys.argv[4]
    fw = open('signature_matching_report.txt', 'w')
    batch_size = 32

    model_triplet = load_model_triplet(model_triplet_path, device='/gpu:0') 
    anchore_network = model_triplet['anchore_network']
    feature_size = anchore_network.get_shape()[1]
    # load prediction {which returns 0 for match and 1 for mis-match}
    model_bin = load_model_bin(model_binary_path, feature_size=feature_size, device='/cpu:0')

    binary_gen = BinaryGenerator(cheque_path, finicale_path, train_valid_split_percent=0.75)
    
    process_time_list = []
    labels_list = []
    predict_data = None
    for img_path_list in binary_gen.valid_data(batch_size=batch_size):
        start_time  = time.time()
        anchore_img_list, test_img_list, labels = img_path_list

        if predict_data is None:
            predict_data = np.round(predict(anchore_img_list, test_img_list, model_triplet, model_bin), 2)
            end_time  = time.time()
        else:
            r = np.round(predict(anchore_img_list, test_img_list, model_triplet, model_bin), 2)
            end_time  = time.time()
            predict_data = np.vstack((predict_data, r))

        process_time_list.append(end_time - start_time)
        labels_list += labels
    report = classification_report(labels_list, np.argmax(predict_data, axis=1))
    conf_mat = confusion_matrix(labels_list, np.argmax(predict_data, axis=1))
    print report
    print "---------------------------------------------------------------------------------"
    print conf_mat
    print "----------------------------------------------------------------------------------"
    print "process time taken:", np.mean(process_time_list)
    fw.write("BATCH SIZE" + str(batch_size))
    fw.write(report)
    fw.write("---------------------------------------------------------------------------------\n")
    fw.write(np.array_str(conf_mat))
    fw.write("---------------------------------------------------------------------------------\n")
    fw.write("process time taken:" + str(np.mean(process_time_list)))
    fw.close()