# RUN - python main.py [path to data dir for triplet] [path to data dir for binary classification] [path to data dir for binary classification accuracy test] [path to forge-genuin dir] [path to forge-genuin dir test] [path to model save dir]
import os, sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from load_data import load_signature_triplet_path, load_signature_binary_path
from network import build_triplet_network, build_binary_classifier_network
from layers.core import input_data
from triplet_loss import triplet_loss
import tensorflow as tf
from utils import fit_image_into_frame, test_create_dir

def sgd(loss, lr=0.001):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
	gvs = optimizer.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_value(grad, -1, +1), var) for grad, var in gvs]
	return optimizer.apply_gradients(capped_gvs)
    # return tf.train.GradientDescentOptimizer(lr).minimize(loss)

def read_img(img_path):
    img = cv2.imread(img_path, 0)
    img = fit_image_into_frame(img, frame_size=(224, 224, 1), random_fill=False, fill_color=255, mode='fit')
    return img

def load_triplet_img_from_path(batch_of_path):
    anchore_img_list = []
    positive_img_list = []
    negative_img_list = []
    for anchore_img_path, positive_img_path, negative_img_path in batch_of_path:
        anchore_img = read_img(anchore_img_path)
        positive_img = read_img(positive_img_path)
        negative_img = read_img(negative_img_path)
        anchore_img_list.append(anchore_img)
        positive_img_list.append(positive_img)
        negative_img_list.append(negative_img)
    return anchore_img_list, positive_img_list, negative_img_list

def load_binary_img_from_path(batch_of_path):
    anchore_img_list = []
    test_img_list = []
    labels = []
    for anchore_img_path, test_img_path, label in batch_of_path:
        anchore_img = read_img(anchore_img_path)
        test_img = read_img(test_img_path)
        anchore_img_list.append(anchore_img)
        test_img_list.append(test_img)
        labels.append(label)
    return anchore_img_list, test_img_list, labels

if __name__ == '__main__':
    data_dir_path_for_triplet = sys.argv[1]
    data_dir_path_for_classification = sys.argv[2]
    data_dir_path_for_classification_test = sys.argv[3]
    data_dir_path_forge_genuin = sys.argv[4]
    data_dir_path_forge_genuin_test = sys.argv[5]
    model_checkpoint_dir = sys.argv[6]

    avg_loss_epoch = []
    avg_diff_sp_sn = []
    avg_loss_epoch_test = []
    num_epoch = 200
    display_step = 10
    batch_size = 16
    # image shape is 224x224x1
    anchore_input = input_data(shape=(224, 224, 1))
    print anchore_input.name
    positive_input = input_data(shape=(224, 224, 1))
    print positive_input.name
    negative_input = input_data(shape=(224, 224, 1))
    print negative_input.name
    dropout_prob = tf.placeholder(tf.float32)
    # create triplet network
    anchore_network, positive_network, negative_network = build_triplet_network(anchore_input, positive_input, negative_input, dropout_prob)

    # get triplet loss
    loss, similar_l2_norm, different_l2_norm = triplet_loss(anchore_network, positive_network, negative_network)

    # train network with SGD
    train_step = sgd(loss, lr=0.0005)

    saver = tf.train.Saver(max_to_keep=None)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    for epoch in xrange(num_epoch):
        total_batch = 0
        loss_list = []
        loss_list_test = []
        diff_list = []
        # for img_path_list in load_signature_triplet_path(data_dir_path_for_triplet, forge_genuine_path=data_dir_path_forge_genuin, batch_size=batch_size):
        for img_path_list in load_signature_triplet_path(data_dir_path_for_triplet, forge_genuine_path=data_dir_path_forge_genuin, forge_genuine_path_sample_size=40, batch_size=batch_size):
            anchore_img_list, positive_img_list, negative_img_list = load_triplet_img_from_path(img_path_list)
            loss_data, sp, sn , _ = sess.run([loss, similar_l2_norm, different_l2_norm, train_step], feed_dict={
                anchore_input: np.asarray(anchore_img_list), 
                positive_input: np.asarray(positive_img_list), 
                negative_input: np.asarray(negative_img_list),
                dropout_prob: 0.5})
            loss_list.append(loss_data)
            diff_list.append(np.mean(sn - sp))
            total_batch += 1
            if total_batch%display_step == 0:
                print 'Train loss', np.mean(loss_list), np.mean(sn - sp), np.mean(sp), np.mean(sn), 'at epoch', epoch, 'on batch', total_batch

        total_batch = 0
        for img_path_list in load_signature_triplet_path(data_dir_path_for_classification, forge_genuine_path=data_dir_path_forge_genuin, forge_genuine_path_sample_size=40, batch_size=batch_size):
            anchore_img_list, positive_img_list, negative_img_list = load_triplet_img_from_path(img_path_list)
            loss_data = sess.run([loss], feed_dict={
                anchore_input: np.asarray(anchore_img_list), 
                positive_input: np.asarray(positive_img_list), 
                negative_input: np.asarray(negative_img_list),
                dropout_prob: 1.0})
            loss_list_test.append(loss_data)
            total_batch += 1 

            if total_batch%display_step == 0:
                print 'Train loss', np.mean(loss_list_test), 'at epoch', epoch, 'on batch', total_batch

        # save plot of loss vs num_of_epoch graph
        avg_loss_epoch.append(np.mean(loss_list))
        avg_loss_epoch_test.append(np.mean(loss_list_test))
        plt.plot(range(len(avg_loss_epoch_test)), avg_loss_epoch_test)
        plt.plot(range(len(avg_loss_epoch)), avg_loss_epoch)
        plt.savefig('plots/TripletLoss_vs_NumIteration.png')
        plt.close()
        
        avg_diff_sp_sn.append(np.mean(diff_list))
        
        # save plot of diff vs num_of_epoch graph
        plt.plot(range(len(avg_diff_sp_sn)), avg_diff_sp_sn)
        plt.savefig('plots/Diff_Sp_Sn_vs_NumIteration.png')
        plt.close()

        # saving model
        print "saveing model ..."
        # test if directory exist, else create epoch dir
        epoch_checkpoint_dir = os.path.join(model_checkpoint_dir, str(epoch).zfill(5))
        test_create_dir(epoch_checkpoint_dir)
        saver.save(sess, os.path.join(epoch_checkpoint_dir, 'model'))

    # create binary classification network.
    avg_loss_epoch = []
    avg_accuracy_epoch = []

    num_epoch = 150
    display_step = 10
    batch_size = 32

    feature_size = anchore_network.get_shape()[1]
    anchore_input_bin = input_data(shape=(None, feature_size))
    test_input_bin = input_data(shape=(None, feature_size))
    labels_bin = tf.placeholder(tf.int64, shape=[None, ])
    dropout_prob_bin = tf.placeholder(tf.float32)

    # create triplet network
    predict_y, loss = build_binary_classifier_network(anchore_input_bin, test_input_bin, labels_bin, dropout_prob_bin)

    correct_prediction = tf.equal(labels_bin, tf.argmax(predict_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train network with SGD
    train_step_bin = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    saver_bin = tf.train.Saver(max_to_keep=None)
    sess_bin = tf.Session()
    init_bin = tf.initialize_all_variables()
    sess_bin.run(init_bin)
    for epoch in xrange(num_epoch):
        total_batch = 0
        loss_list = []
        accuracy_list = []
        for img_path_list in load_signature_binary_path(data_dir_path_for_classification, forge_genuine_path=data_dir_path_forge_genuin, batch_size=batch_size):
            anchore_img_list, test_img_list, labels = load_binary_img_from_path(img_path_list)
            
            anchore_img_features = sess.run(anchore_network,  feed_dict={anchore_input: np.asarray(anchore_img_list), dropout_prob: 1.0})
            test_img_features = sess.run(anchore_network,  feed_dict={anchore_input: np.asarray(test_img_list), dropout_prob: 1.0})

            loss_data , _ = sess_bin.run([loss, train_step_bin], feed_dict={
                anchore_input_bin: np.asarray(anchore_img_features), 
                test_input_bin: np.asarray(test_img_features), labels_bin: labels, dropout_prob_bin: 0.5})


            loss_list.append(loss_data)
            total_batch += 1

            if total_batch%display_step == 0:
                print 'loss', np.mean(loss_list), 'at epoch', epoch, 'on batch', total_batch


        for img_path_list in load_signature_binary_path(data_dir_path_for_classification_test, forge_genuine_path=data_dir_path_forge_genuin_test, batch_size=batch_size):
            anchore_img_list, test_img_list, labels = load_binary_img_from_path(img_path_list)
            anchore_img_features = sess.run(anchore_network,  feed_dict={anchore_input: np.asarray(anchore_img_list), dropout_prob: 1.0})
            test_img_features = sess.run(anchore_network,  feed_dict={anchore_input: np.asarray(test_img_list), dropout_prob: 1.0})
            accuracy_list.append(sess_bin.run(accuracy, {anchore_input_bin: anchore_img_features, test_input_bin: test_img_features, labels_bin: labels, dropout_prob_bin:1.0}))

        avg_loss_epoch.append(np.mean(loss_list))
        avg_acc = np.mean(accuracy_list)
        print "accuracy at ", epoch, 'epoch', avg_acc
        avg_accuracy_epoch.append(avg_acc)

        plt.plot(range(len(avg_accuracy_epoch)), avg_accuracy_epoch)
        plt.plot(range(len(avg_loss_epoch)), avg_loss_epoch)
        plt.savefig('plots/Loss_n_Accuracy_vs_NumIteration.png')
        plt.close()

        # # saving model
        print "saveing model ..."
        # test if directory exist, else create epoch dir
        epoch_checkpoint_dir = os.path.join(model_checkpoint_dir, str(epoch).zfill(5)+'_BIN')
        test_create_dir(epoch_checkpoint_dir)
        saver.save(sess_bin, os.path.join(epoch_checkpoint_dir, 'model'))

