import os, sys
import numpy as np
import tensorflow as tf
import load_data
from network import build_filter_network
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from shutil import copyfile

def test_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

if __name__ == '__main__':
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    model_checkpoint_dir = 'model'

    # create placeholder for variables
    input_t = tf.placeholder(tf.float32, [None, 128, 128, 1])
    labels_t = tf.placeholder(tf.int64, shape=[None, ])
    dropout_prob_t = tf.placeholder(tf.float32)

    predict, _ = build_filter_network(input_t, labels_t, dropout_prob_t)

    saver = tf.train.Saver()
    sess = tf.Session()
    tf.initialize_all_variables()
    saver.restore(sess, '/home/arya_01/AxisProject/AxisFilter/model/00247_FILTER/model')
    # sess.run(init)
   
    with sess.as_default():
        for imgs, labels, filelist in load_data.read_img_test(test_dir, balance_class=False):
            predict_data = predict.eval(feed_dict={
                input_t: np.asarray(imgs),
                dropout_prob_t: 1.0})
            # for d, fp, ll, pp in zip(np.logical_xor(np.argmax(predict_data, axis=1) == 1, np.asarray(labels) == 1), filelist, labels, np.argmax(predict_data, axis=1)):
            #     if d == True:
            #         fname = fp.split('/')[-1]
            #         if ll == 0:
            #             copyfile(fp, os.path.join('tmp/no', fname))
            #         else:
            #             copyfile(fp, os.path.join('tmp/yes', fname))
            #         print fp, ll, pp

            for pp, ll, fp in zip(predict_data, np.asarray(labels), filelist):
                fname = fp.split('/')[-1]
                if ll == 0 and not(pp[0] >= 0.20):
                    copyfile(fp, os.path.join('tmp/no', fname))
                    print ll, pp[ll]
                if ll == 1 and not(pp[1] >= 0.8):
                    copyfile(fp, os.path.join('tmp/yes', fname))
                    print ll, pp[ll]
                # if pp[0] >= 0.4 and ll != 0:
                #     copyfile(fp, os.path.join('tmp/yes', fname))
                # elif pp[1] >= 0.6 and ll != 1:
                #     copyfile(fp, os.path.join('tmp/no', fname))
            # for d, fp, ll, pp in zip(np.logical_xor(np.argmax(predict_data<0.4, axis=1), np.asarray(labels) == 1), filelist, labels, np.argmax(predict_data, axis=1)):
            #     if d == True:
            #         fname = fp.split('/')[-1]
            #         if ll == 0:
            #             copyfile(fp, os.path.join('tmp/no', fname))
            #         else:
            #             copyfile(fp, os.path.join('tmp/yes', fname))
            #         print fp, ll, pp
