# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

from . import variables as vs
from . import activations
from . import initializations
from . import utils


def conv_2d(incoming, nb_filter, filter_size, strides=1, padding='same',
            activation='linear', bias=True, weights_init='uniform_scaling',
            bias_init='zeros', regularizer=None, weight_decay=0.001,
            trainable=True, restore=True, name="Conv2D"):
    """ Convolution 2D.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: 'int` or list of `ints`. Size of filters.
        strides: 'int` or list of `ints`. Strides of conv operation.
            Default: [1 1 1 1].
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model
        name: A name for this layer (optional). Default: 'Conv2D'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        b: `Variable`. Variable representing biases.

    """
    assert padding in ['same', 'valid', 'SAME', 'VALID'], \
        "Padding must be same' or 'valid'"

    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
    filter_size = utils.autoformat_filter_conv2d(filter_size,
                                                 input_shape[-1],
                                                 nb_filter)
    strides = utils.autoformat_kernel_2d(strides)
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(name) as scope:
        scope = scope.name + '/'
    # with tf.name_scope(name) as scope:
        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        
        W = vs.variable(scope + 'W', shape=filter_size,
                        regularizer=W_regul, initializer=W_init,
                        trainable=trainable, restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, W)
        b = None
        if bias:
            b_init = initializations.get(bias_init)()
            b = vs.variable(scope + 'b', shape=nb_filter,
                            initializer=b_init, trainable=trainable, restore=restore)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, b)

        inference = tf.nn.conv2d(incoming, W, strides, padding)
        if b: inference = tf.nn.bias_add(inference, b)

        if isinstance(activation, str):
            inference = activations.get(activation)(inference)
        elif hasattr(activation, '__call__'):
            inference = activation(inference)
        else:
            raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    return inference

def max_pool_2d(incoming, kernel_size, strides=None, padding='same',
                name="MaxPool2D"):
    """ Max Pooling 2D.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, pooled height, pooled width, in_channels].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        kernel_size: 'int` or list of `ints`. Pooling kernel size.
        strides: 'int` or list of `ints`. Strides of conv operation.
            Default: same as kernel_size.
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'MaxPool2D'.

    Attributes:
        scope: `Scope`. This layer scope.

    """
    assert padding in ['same', 'valid', 'SAME', 'VALID'], \
        "Padding must be same' or 'valid'"

    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    kernel = utils.autoformat_kernel_2d(kernel_size)
    strides = utils.autoformat_kernel_2d(strides) if strides else kernel
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(name) as scope:
        scope = scope.name + '/'
    # with tf.name_scope(name) as scope:
        inference = tf.nn.max_pool(incoming, kernel, strides, padding)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    return inference


def avg_pool_2d(incoming, kernel_size, strides=None, padding='same',
                name="AvgPool2D"):
    """ Average Pooling 2D.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, pooled height, pooled width, in_channels].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        kernel_size: 'int` or list of `ints`. Pooling kernel size.
        strides: 'int` or list of `ints`. Strides of conv operation.
            Default: same as kernel_size.
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'AvgPool2D'.

    Attributes:
        scope: `Scope`. This layer scope.

    """
    assert padding in ['same', 'valid', 'SAME', 'VALID'], \
        "Padding must be same' or 'valid'"

    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    kernel = utils.autoformat_kernel_2d(kernel_size)
    strides = utils.autoformat_kernel_2d(strides) if strides else kernel
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(name) as scope:
        scope = scope.name + '/'
    # with tf.name_scope(name) as scope:
        inference = tf.nn.avg_pool(incoming, kernel, strides, padding)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    return inference
