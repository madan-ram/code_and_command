import tensorflow as tf
from layers.core import dropout, fully_connected
from layers.conv import conv_2d, max_pool_2d
from tensorflow.python.ops import rnn_cell, rnn
from layers.normalization import local_response_normalization
from math import sqrt
import pprint

def deepnet(shape=None, fan_out=1, dtype=tf.float32):
    stddev = sqrt(2. / (fan_out))
    return tf.truncated_normal_initializer(stddev=stddev)

def get_conv2d(input, n_filter, filter_size, max_pool_size, name, strides=1, trainable=True, padding='same'):
    if type(filter_size) == tuple:
        weights = deepnet(fan_out=(n_filter*filter_size[0]*filter_size[1])//max_pool_size)
    else:
        weights = deepnet(fan_out=(n_filter*filter_size*filter_size)//max_pool_size)
    network = conv_2d(input, n_filter, filter_size, strides=strides, activation='relu', name=name, weights_init=weights, padding=padding)
    return network

def infer_network(input, dropout_prob):
    name_net_dict = {}
    network = get_conv2d(input, 64, 3, max_pool_size=1, name='conv_a1', strides=2)
    network = get_conv2d(network, 64, 3, max_pool_size=1, name='conv_a2')
    network = max_pool_2d(network, 2)
    name_net_dict['conva'] = network

    network = get_conv2d(network, 128, 3, max_pool_size=1, name='conv_b1')
    network = get_conv2d(network, 128, 3, max_pool_size=2, name='conv_b2')
    network = max_pool_2d(network, 2)
    name_net_dict['convb'] = network

    network = get_conv2d(network, 128, 3, max_pool_size=1, name='conv_c1')
    network = get_conv2d(network, 128, 3, max_pool_size=1, name='conv_c2')
    network = max_pool_2d(network, 2)
    name_net_dict['convc'] = network

    network = get_conv2d(network, 256, 3, max_pool_size=1, name='conv_d1')
    network = get_conv2d(network, 256, 3, max_pool_size=1, name='conv_d2')
    network = max_pool_2d(network, 2)
    name_net_dict['convd'] = network

    network = get_conv2d(network, 512, 3, max_pool_size=1, name='conv_e1')
    network = get_conv2d(network, 512, 3, max_pool_size=1, name='conv_e2')
    network = max_pool_2d(network, 2)
    name_net_dict['conve'] = network

    network_feature_sequence = get_conv2d(network, 512, (2, 1), max_pool_size=1, name='feature_sequence', padding='valid')
    name_net_dict['feature_sequence'] = network_feature_sequence
    
    network = get_conv2d(network, 512, (1, 3), max_pool_size=1, name='conv_f1', padding='valid')
    network = get_conv2d(network, 512, (1, 3), max_pool_size=1, name='conv_f2', padding='valid')
    network = max_pool_2d(network, (1, 3))
    name_net_dict['convf'] = network

    network = get_conv2d(network, 512, (1, 3), max_pool_size=1, name='conv_g1', padding='valid')
    name_net_dict['convg'] = network
    
    print network.get_shape(), 'CONV_g4 shape'
    print network_feature_sequence.get_shape(), 'network_feature_sequence'

    network = fully_connected(network, 4096, activation='relu', name='feature_layer1')
    with tf.device('/cpu:0'):
        network = dropout(network, dropout_prob)

    return network, name_net_dict

def create_lstm_seq(input, length_labels, units_size, batch_size=32, num_rnn_layers=None, out_activation_func=None):
    """
    LSTM with input of (batch_size, width as n_steps, and height*feature size)
    """
    batch_size_d, height_d, width_d, feature_size_d = input.get_shape()
    height, width, feature_size = height_d.value, width_d.value, feature_size_d.value

    new_feature_size = feature_size*height
    n_steps = width

    input = tf.transpose(input, [0, 2, 1, 3])
    input = tf.reshape(input, [-1, n_steps, new_feature_size])
    input = tf.transpose(input, [1, 0, 2])

    input = tf.reshape(input, [-1, new_feature_size])
    inputs = tf.split(0, n_steps, input)
    print 'New input shape', inputs[0].get_shape()
    print 'number of steps', len(inputs)

    lstm_fw_cell = rnn_cell.BasicLSTMCell(units_size, state_is_tuple=True)
    lstm_bw_cell = rnn_cell.BasicLSTMCell(units_size, state_is_tuple=True)
    if num_rnn_layers is not None:
        lstm_fw_cell = rnn_cell.MultiRNNCell([lstm_fw_cell] * num_rnn_layers, state_is_tuple=True)
        lstm_bw_cell = rnn_cell.MultiRNNCell([lstm_bw_cell] * num_rnn_layers, state_is_tuple=True)

    try:
        outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, sequence_length=None,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs,
                                        dtype=tf.float32)


    if out_activation_func is not None:
        outputs = [out_activation_func(o) for o in outputs]

    print 'New output shape', tf.pack(outputs).get_shape()

    return locals()

def build_multi_layer(input, num_multi_classes, sequence_length, dropout_prob, batch_size=32):

    multi_labels = tf.placeholder(shape=(None, num_multi_classes, 2), dtype=tf.float32, name='multi_labels')
    sequence_length_label = tf.placeholder(shape=(None, ), dtype=tf.int64, name='sequence_length_label')

    cnn_network, name_net_dict = infer_network(input, dropout_prob)

    predict_multi_list = []
    loss =  0.0
    for m_id in xrange(num_multi_classes):

        network = fully_connected(cnn_network, 1024, activation='relu', name='full_'+str(m_id+1)+'_1')
        with tf.device('/cpu:0'):
            network = dropout(network, dropout_prob)

        network = fully_connected(network, 2, activation='relu', name='full_'+str(m_id+1)+'_2')
        predict_network = tf.nn.softmax(network, name='predict_'+str(m_id+1))

        predict_multi_list.append(predict_network)
        loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network, multi_labels[:, m_id, :]))

    network = fully_connected(cnn_network, 1024, activation='relu', name='seq_len_full_1')
    with tf.device('/cpu:0'):
        network = dropout(network, dropout_prob)

    network = fully_connected(network, sequence_length+1, activation='relu', name='seq_len_full_2')
    sequence_length_predict = tf.nn.softmax(network, name='length_predict')

    length_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(network, sequence_length_label))
    loss += length_loss

    predict_multi = tf.pack(predict_multi_list)

    return locals()

def build_network(input, feature_sequence, N, num_classes, dropout_prob, batch_size=32):
    """
    LSTM based network without CTC
    """
    RNN_UNIT_SIZE = 256
    NUM_RNN_LAYER = 10

    number_indices = tf.placeholder(shape=(None, 2), dtype=tf.int64, name='number_indices')
    number_values = tf.placeholder(shape=(None, ), dtype=tf.int32, name='number_values')
    length_labels = tf.placeholder(shape=(None, ), dtype=tf.int32, name='length_labels')

    # use rnn to model sequence over all data
    rnn_model = create_lstm_seq(feature_sequence, length_labels, RNN_UNIT_SIZE, num_rnn_layers=NUM_RNN_LAYER, batch_size=batch_size, out_activation_func=tf.nn.relu)

    number_network_list = []
    number_predict_list = []
    for _id, number_network in enumerate(rnn_model['outputs']):

        number_network = fully_connected(number_network, num_classes, activation='relu', name='number_full'+str(_id+1)+'_1')
        number_predict = tf.nn.softmax(number_network, name='number_predict'+str(_id+1))

        number_network_list.append(number_network)
        number_predict_list.append(number_predict)

    t = len(number_predict_list)
    b = batch_size
    n = num_classes

    number_labels = tf.SparseTensor(indices=number_indices, values=number_values, shape=(t, b))
    loss = tf.reduce_mean(tf.nn.ctc_loss(number_network_list, number_labels, length_labels, preprocess_collapse_repeated=False, ctc_merge_repeated=False))
    
    number_predict_list = tf.pack(number_predict_list)
    del _id
    return locals()

if __name__ == '__main__':
    from layers.core import input_data
    input = input_data(shape=(None, 128, 896, 1), name='input')
    dropout_prob = tf.placeholder(tf.float32)

    model = build_multi_layer(input, 50, 7, dropout_prob, batch_size=32)
    feature_sequence = model['name_net_dict']['feature_sequence']

    model = build_network(input, feature_sequence, 11, 10, dropout_prob)
    print model