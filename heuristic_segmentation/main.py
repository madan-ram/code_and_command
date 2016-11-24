import os, sys
import numpy as np
import tensorflow as tf
import load_data
from network import build_filter_network
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def sgd(loss, lr=0.001):
    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1, +1), var) for grad, var in gvs]
    return optimizer.apply_gradients(capped_gvs)

def test_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

if __name__ == '__main__':
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    model_checkpoint_dir = 'model'
    display_step = 10
    num_epoch = 250

    # create placeholder for variables
    input_t = tf.placeholder(tf.float32, [None, 128, 128, 1])
    labels_t = tf.placeholder(tf.int64, shape=[None, ])
    dropout_prob_t = tf.placeholder(tf.float32)

    predict, loss = build_filter_network(input_t, labels_t, dropout_prob_t)

    train_step = sgd(loss, lr=0.00001)
    correct_prediction = tf.equal(labels_t, tf.argmax(predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(max_to_keep=None)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    avg_train_loss_epoch = []
    avg_test_loss_epoch = []
    avg_test_acc = []
    max_acc_epoch = -1
    max_acc = -1
    for epoch in xrange(num_epoch):
        loss_list = []
        batch_id = 0
        for imgs, labels in load_data.read_img(train_dir, batch_size=32):
            loss_data, _ = sess.run([loss, train_step], feed_dict={
                input_t: np.asarray(imgs), 
                labels_t: np.asarray(labels),
                dropout_prob_t: 0.5})

            batch_id += 1
            loss_list.append(loss_data)

            if batch_id%display_step == 0:
                print 'loss', np.mean(loss_list), 'at epoch', epoch, 'on batch', batch_id
 
        avg_train_loss_epoch.append(np.mean(loss_list))


        # get the test loss
        acc_list = []
        loss_list = []
        for imgs, labels in load_data.read_img(test_dir, balance_class=False):
            loss_data = sess.run([loss], feed_dict={
                input_t: np.asarray(imgs), 
                labels_t: np.asarray(labels),
                dropout_prob_t: 1.0})
            accuracy_data = sess.run([accuracy], feed_dict={
                input_t: np.asarray(imgs), 
                labels_t: np.asarray(labels),
                dropout_prob_t: 1.0})
            loss_list.append(loss_data)
            acc_list.append(accuracy_data)
 

        if max_acc <= np.mean(acc_list):
            max_acc = np.mean(acc_list)
            max_acc_epoch = epoch

        print "average test loss", np.mean(loss_list)
        avg_test_loss_epoch.append(np.mean(loss_list))

        print "average test accuracy", np.mean(acc_list)
        avg_test_acc.append(np.mean(acc_list))

        print 'MAX ACC EPOCH', max_acc_epoch


        plt.plot(range(len(avg_test_acc)), avg_test_acc, 'r-')
        plt.plot(range(len(avg_train_loss_epoch)), avg_train_loss_epoch, 'g-')
        plt.plot(range(len(avg_test_loss_epoch)), avg_test_loss_epoch, 'b-')
        plt.savefig('plots/Loss_n_Accuracy_vs_NumIteration.png')
        plt.close()


        print "saveing model ..."
        # test if directory exist, else create epoch dir
        epoch_checkpoint_dir = os.path.join(model_checkpoint_dir, str(epoch).zfill(5)+'_FILTER')
        test_create_dir(epoch_checkpoint_dir)
        saver.save(sess, os.path.join(epoch_checkpoint_dir, 'model'))


