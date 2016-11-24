# RUN - python main.py [path to cheque data dir] [path to finicale data dir] [model save path]
# RUN SAMPLE -  python train_bin.py /home/aipocuser/Axis_Project/code/signature_training/output/cheque /home/aipocuser/Axis_Project/code/signature_training/output/finicale ./model
import os, sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from load_data import AmountInWordGenerator, AmountInWordMultiGenerator
from network_adv_ctc import build_network, build_multi_layer
from layers.core import input_data
import tensorflow as tf
from utils import fit_image_into_frame, test_create_dir
from scipy.sparse import coo_matrix

def sgd(loss, lr=0.001):
    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1, +1), var) for grad, var in gvs if grad is not None]
    return optimizer.apply_gradients(capped_gvs)
    # return tf.train.GradientDescentOptimizer(lr).minimize(loss)

def guarantee_initialized_variables(session, list_of_variables = None):
    if list_of_variables is None:
        list_of_variables = tf.all_variables()
    uninitialized_variables = list(tf.get_variable(name) for name in
                                   session.run(tf.report_uninitialized_variables(list_of_variables)))
    session.run(tf.initialize_variables(uninitialized_variables))
    return unintialized_variables

def dence_to_sparse(A, mask):
    A = np.argmax(A, axis=2)
    A = A.T
    mask = mask.T
    A = (A + 1) * mask
    a = coo_matrix(A)
    a.data = a.data - 1
    indices = np.asarray([a.row, a.col]).T
    values = a.data
    shape = a.shape
    return indices, values, shape

def get_exact_match_acc(prediction, actual_result, mask):
    correct_prediction = []
    prediction = np.argmax(prediction, axis=2)
    actual_result = np.argmax(actual_result, axis=2)
    p_mask = (prediction == actual_result)
    n_chars, batch_size = prediction.shape
    for b in xrange(batch_size):
        gather_list = []
        for n in xrange(n_chars):
            if mask[n][b]:
                gather_list.append(p_mask[n][b])
        if all(gather_list):
            correct_prediction.append(1.0)
        else:
            correct_prediction.append(0.0)
    return np.mean(correct_prediction)


if __name__ == '__main__':
    n_chars = int(sys.argv[1])
    multi_lables = ["crore", "lakh", "thousand", "hundred", "ninety", "eighty", "seventy", "sixty", 
        "fifty", "forty", "thirty", "twenty", "nineteen", "eighteen", "seventeen", "sixteen", "fifteen", 
        "fourteen", "thirteen", "twelve", "eleven", "ten", "nine", "eight" , "seven", "six", "five", "four", 
        "three", "two", "one", "zero"]
    display_step = 10
    input_image_shape = (128, 896, 1)
    num_multi_classes = len(multi_lables)
    amount_in_word_multi_gen = AmountInWordMultiGenerator(n_chars, multi_lables, frame_shape=input_image_shape, train_valid_split_percent=0.75)

    # Train network to perform multi-label classification
    feature_sequence_network = None
    input = None
    dropout_prob = None
    batch_size = 32
    num_epoch = 2
    g = tf.Graph()
    sess = tf.Session(graph=g, config=tf.ConfigProto(
    intra_op_parallelism_threads=4))
    with g.as_default():

        input = input_data(shape=(None, input_image_shape[0], input_image_shape[1], input_image_shape[2]), name='input')
        dropout_prob = tf.placeholder(tf.float32)

        multi_label_model = build_multi_layer(input, num_multi_classes, n_chars, dropout_prob, batch_size=batch_size)
        sequence_length_label = multi_label_model['sequence_length_label']
        sequence_length_predict = multi_label_model['sequence_length_predict']
        loss = multi_label_model['loss']
        multi_labels = multi_label_model['multi_labels']
        predict_multi = multi_label_model['predict_multi']
        feature_sequence_network = multi_label_model['name_net_dict']['feature_sequence']
        train_step = sgd(loss, lr=0.0001)

        avg_loss_epoch = []
        avg_test_loss_epoch = []
        avg_length_accuracy_epoch = []
        avg_multi_accuracy_epoch = []

        correct_prediction = tf.equal(sequence_length_label, tf.argmax(sequence_length_predict, 1))
        length_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        correct_prediction = tf.equal(tf.argmax(predict_multi, dimension=2), tf.argmax(multi_labels, dimension=2))
        multi_labels_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(max_to_keep=None)
        init = tf.initialize_all_variables()
        sess.run(init)
        for epoch in xrange(num_epoch):
            loss_list = []
            length_accuracy_list = []
            multi_labels_accuracy_list = []

            total_batch = 0
            for data in amount_in_word_multi_gen.train_data(batch_size=batch_size):
                input_list_d, length_label_list_d, multi_labels_d = data
                loss_data, length_accuracy_data, multi_labels_accuracy_data, _ = sess.run([loss, length_accuracy, multi_labels_accuracy, train_step], feed_dict={
                    input: np.asarray(input_list_d, dtype=np.float32),
                    sequence_length_label: np.asarray(length_label_list_d, dtype=np.int64),
                    multi_labels: np.asarray(multi_labels_d),
                    dropout_prob: 0.5})

                loss_list.append(loss_data)
                length_accuracy_list.append(length_accuracy_data)
                multi_labels_accuracy_list.append(multi_labels_accuracy_data)

                total_batch += 1
                if total_batch%display_step == 0:
                    print 'Train Avg Loss', np.mean(loss_list), 'Train length acc', np.mean(length_accuracy_list), 'Train multi acc', np.mean(multi_labels_accuracy_list), 'at epoch', epoch, 'on batch', '['+str(total_batch)+'/'+str(amount_in_word_multi_gen.num_train_batch)+']'
                    break

            loss_list = []
            length_accuracy_list = []
            multi_labels_accuracy_list = []

            total_batch = 0
            for data in amount_in_word_multi_gen.valid_data(batch_size=batch_size):
                input_list_d, length_label_list_d, multi_labels_d = data
                loss_data, length_accuracy_data, multi_labels_accuracy_data = sess.run([loss, length_accuracy, multi_labels_accuracy], feed_dict={
                    input: np.asarray(input_list_d, dtype=np.float32),
                    sequence_length_label: np.asarray(length_label_list_d, dtype=np.int64),
                    multi_labels: np.asarray(multi_labels_d),
                    dropout_prob: 1.0})

                loss_list.append(loss_data)
                length_accuracy_list.append(length_accuracy_data)
                multi_labels_accuracy_list.append(multi_labels_accuracy_data)

                total_batch += 1
                if total_batch%display_step == 0:
                    print 'Test Avg Loss', np.mean(loss_list), 'at epoch', epoch, 'on batch', '['+str(total_batch)+'/'+str(amount_in_word_multi_gen.num_valid_batch)+']'
                    break

            print 'Test length acc', np.mean(length_accuracy_list), 'Test multi acc', np.mean(multi_labels_accuracy_list)


    # Train network to perform CTC based number classification
    number_of_class = 10
    # for CTC we need num_classes+1 for blank label
    number_of_class += 1

    batch_size = 32
    num_epoch = 2
    with g.as_default():

        # input = input_data(shape=(None, input_image_shape[0], input_image_shape[1], input_image_shape[2]), name='input')
        # dropout_prob = tf.placeholder(tf.float32)

        model = build_network(input, feature_sequence_network, n_chars, number_of_class, dropout_prob, batch_size=32)
        number_indices  = model['number_indices']
        number_values = model['number_values']
        length_labels  = model['length_labels']
        # get overall loss
        loss = model['loss']
        number_predict = model['number_predict_list']

        train_step = sgd(loss, lr=0.0001)
        # initialize all uninitialized variables
        guarantee_initialized_variables(sess)
        amount_in_word_gen = AmountInWordGenerator(n_chars, frame_shape=input_image_shape, num_classes=number_of_class, train_valid_split_percent=0.75)
        saver = tf.train.Saver(max_to_keep=None)
        for epoch in xrange(num_epoch):
            total_batch = 0
            loss_list = []
            loss_list_test = []
            length_accuracy_data_list = []
            S_accuracy_data_list = []
            S_accuracy_data_each_list = []
            for data in amount_in_word_gen.train_data(batch_size=batch_size):
                input_list_d, length_label_list_d, number_labels_d, mask_matrix_d = data
                indices, values, shape = dence_to_sparse(number_labels_d, mask_matrix_d)
                loss_data, number_predict_data, _ = sess.run([loss, number_predict, train_step], feed_dict={
                    input: np.asarray(input_list_d, dtype=np.float32),
                    length_labels: np.asarray(length_label_list_d, dtype=np.int64),
                    number_indices: indices,
                    number_values: values,
                    dropout_prob: 0.5})


                loss_list.append(loss_data)
                total_batch += 1
                if total_batch%display_step == 0:
                    print 'Train Avg  Loss', np.mean(loss_list), 'Train Loss', loss_data, 'at epoch', epoch, 'on batch', '['+str(total_batch)+'/'+str(amount_in_word_gen.num_train_batch)+']'
                    if total_batch%100 == 0:
                        print np.argmax(number_predict_data, axis=2).T[:2]
                        print np.argmax(number_labels_d, axis=2).T[:2]

            total_batch = 0
            for data in amount_in_word_gen.valid_data(batch_size=batch_size):
                input_list_d, length_label_list_d, number_labels_d, mask_matrix_d = data
                indices, values, shape = dence_to_sparse(number_labels_d, mask_matrix_d)
                loss_data, number_predict_data = sess.run([loss, number_predict], feed_dict={
                    input: np.asarray(input_list_d, dtype=np.float32),
                    length_labels: np.asarray(length_label_list_d, dtype=np.int64),
                    number_indices: indices,
                    number_values: values,
                    dropout_prob: 1.0})

                loss_list_test.append(loss_data)
                total_batch += 1
                if total_batch%display_step == 0:
                    print 'Test Avg Loss', np.mean(loss_list_test), 'Test Loss', loss_data, 'at epoch', epoch, 'on batch', '['+str(total_batch)+'/'+str(amount_in_word_gen.num_valid_batch)+']'
            
            avg_loss_epoch.append(np.mean(loss_list))
            avg_test_loss_epoch.append(np.mean(loss_list_test))

            plt.plot(range(len(avg_test_loss_epoch)), avg_test_loss_epoch, 'y')
            plt.plot(range(len(avg_loss_epoch)), avg_loss_epoch, 'r')
            plt.savefig('plots/Loss_test_n_train_vs_NumIteration_ctc.png')
            plt.close()