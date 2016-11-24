import tensorflow as tf
from layers.core import dropout, fully_connected
from layers.conv import conv_2d, max_pool_2d
from layers.normalization import local_response_normalization
from math import sqrt

def deepnet(shape=None, fan_out=1, dtype=tf.float32):
    """ This initialization gives good performance with deep nets,
    e.g.: VGG-16.

    This method [1]_ was mainly developed with Relu/PRelu activations in mind
    and is known to give good performance with very deep networks which use
    these activations.

    Parameters
    ----------
    shape : tuple or list
        Shape of the weight tensor to sample.

    fan_out : int
        The number of units connected at the output of the current layer.

    Returns
    -------
    float
        Standard deviation of the normal distribution for weight
        initialization.

    References
    ----------
    .. [1] He, K., Zhang, X., Ren, S., and Sun, J. Delving Deep
           into Rectifiers: Surpassing Human-Level Performance
           on ImageNet Classification. ArXiv e-prints, February
           2015.

    Notes
    -----
    The weights are initialized as

    .. math::
        \\sigma = \\sqrt{\\frac{2}{fan_{out}}}

    """
    stddev = sqrt(2. / (fan_out))
    return tf.truncated_normal_initializer(stddev=stddev)

def get_conv2d(input, n_filter, filter_size, max_pool_size, name, strides=1, trainable=True):
    weights = deepnet(fan_out=(n_filter*filter_size*filter_size)//max_pool_size)
    network = conv_2d(input, n_filter, filter_size, strides=strides, activation='relu', name=name, weights_init=weights)
    return network

def infer_network(input, dropout_prob):
    name_net_dict = {}
    
    network = get_conv2d(input, 32, 3, max_pool_size=1, name='conv_a1')
    network = get_conv2d(network, 32, 3, max_pool_size=2, name='conv_a2')
    network = max_pool_2d(network, 2)
    # network = local_response_normalization(network)

    network = get_conv2d(network, 64, 3, max_pool_size=1, name='conv_b1')
    network = get_conv2d(network, 64, 3, max_pool_size=2, name='conv_b2')
    network = max_pool_2d(network, 2)
    # network = local_response_normalization(network)

    network = get_conv2d(network, 96, 3, max_pool_size=1, name='conv_c1')
    network = get_conv2d(network, 96, 3, max_pool_size=1, name='conv_c2')
    network = get_conv2d(network, 96, 3, max_pool_size=1, name='conv_c3')
    network = get_conv2d(network, 96, 3, max_pool_size=2, name='conv_c4')
    network = max_pool_2d(network, 2)
    # network = local_response_normalization(network)

    network = get_conv2d(network, 128, 3, max_pool_size=1, name='conv_d1')
    network = get_conv2d(network, 128, 3, max_pool_size=1, name='conv_d2')
    network = get_conv2d(network, 128, 3, max_pool_size=1, name='conv_d3')
    network = get_conv2d(network, 128, 3, max_pool_size=2, name='conv_d4')
    network = max_pool_2d(network, 2)
    # network = local_response_normalization(network)

    network = get_conv2d(network, 256, 3, max_pool_size=1, name='conv_e1')
    network = get_conv2d(network, 256, 3, max_pool_size=1, name='conv_e2')
    network = get_conv2d(network, 256, 3, max_pool_size=1, name='conv_e3')
    network = get_conv2d(network, 256, 3, max_pool_size=2, name='conv_e4')
    network = max_pool_2d(network, 2)

    # network = get_conv2d(network, 256, 3, max_pool_size=1, name='conv_f1')
    # network = get_conv2d(network, 256, 3, max_pool_size=1, name='conv_f2')
    # network = get_conv2d(network, 256, 3, max_pool_size=1, name='conv_f3')
    # network = get_conv2d(network, 256, 3, max_pool_size=2, name='conv_f4')
    # network = max_pool_2d(network, 2)
    print network.get_shape(), 'CONV_E4 shape'
    # network = local_response_normalization(network)

    network = fully_connected(network, 1024, activation='relu', name='full1')
    network = dropout(network, dropout_prob)
    network = fully_connected(network, 1024, activation='relu', name='full2')
    network = dropout(network, dropout_prob)
    network = fully_connected(network, 2, activation='relu', name='full3')
    return network

def build_filter_network(input, labels, dropout_prob):
    # create anchore images
    with tf.variable_scope('FilterNetwork') as scope:
        network = infer_network(input, dropout_prob)

    predict = tf.nn.softmax(network)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(network, labels))
   
    return predict, loss