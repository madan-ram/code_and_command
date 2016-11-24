# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow.python.training import moving_averages

from . import utils
from . import variables as vs

def local_response_normalization(incoming, depth_radius=5, bias=1.0,
                                 alpha=0.0001, beta=0.75,
                                 name="LocalResponseNormalization"):
    """ Local Response Normalization.

    Input:
        4-D Tensor Layer.

    Output:
        4-D Tensor Layer. (Same dimension as input).

    Arguments:
        incoming: `Tensor`. Incoming Tensor.
        depth_radius: `int`. 0-D.  Half-width of the 1-D normalization window.
            Defaults to 5.
        bias: `float`. An offset (usually positive to avoid dividing by 0).
            Defaults to 1.0.
        alpha: `float`. A scale factor, usually positive. Defaults to 0.0001.
        beta: `float`. An exponent. Defaults to `0.5`.

    """

    with tf.variable_scope(name) as scope:
        scope = scope.name + '/'
    # with tf.name_scope(name) as scope:
        inference = tf.nn.lrn(incoming, depth_radius=depth_radius,
                              bias=bias, alpha=alpha,
                              beta=beta, name=name)

    inference.scope = scope

    return inference
