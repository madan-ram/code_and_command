from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from . import utils
from . import variables as va
from . import activations
from . import initializations
# from tflearn import losses

def get_training_mode():
    """ get_training_mode.

    Returns variable in-use to set training mode.

    Returns:
        A `Variable`, the training mode holder.

    """
    init_training_mode()
    coll = tf.get_collection('is_training')
    return coll[0]


def init_training_mode():
    """  init_training_mode.

    Creates `is_training` variable and its ops if they haven't be created
    yet. This op is required if you are using layers such as dropout or
    batch normalization independently of TFLearn models (DNN or Trainer class).

    """
    # 'is_training' collection stores the training mode variable
    coll = tf.get_collection('is_training')
    if len(coll) == 0:
        tr_var = tf.get_variable(
            "is_training", dtype=tf.bool, shape=[],
            initializer=tf.constant_initializer(False),
            trainable=False)
        tf.add_to_collection('is_training', tr_var)
        # 'is_training_ops' stores the ops to update training mode variable
        a = tf.assign(tr_var, True)
        b = tf.assign(tr_var, False)
        tf.add_to_collection('is_training_ops', a)
        tf.add_to_collection('is_training_ops', b)


def input_data(shape=None, placeholder=None, dtype=tf.float32,
               data_preprocessing=None, data_augmentation=None,
               name="InputData"):
    """ Input Data.

    `input_data` is used as a data entry (placeholder) of a network.
    This placeholder will be feeded with data when training

    This layer is used to keep track of the network inputs, by adding the
    placeholder to INPUTS graphkey. TFLearn training functions may retrieve
    those variables to setup the network training process.

    Input:
        List of `int` (Shape), to create a new placeholder.
            Or
        `Tensor` (Placeholder), to use an existing placeholder.

    Output:
        Placeholder Tensor with given shape.

    Arguments:
        shape: list of `int`. An array or tuple representing input data shape.
            It is required if no placeholder provided. First element should
            be 'None' (representing batch size), if not provided, it will be
            added automatically.
        placeholder: A Placeholder to use for feeding this layer (optional).
            If not specified, a placeholder will be automatically created.
            You can retrieve that placeholder through graph key: 'INPUTS',
            or the 'placeholder' attribute of this function's returned tensor.
        dtype: `tf.type`, Placeholder data type (optional). Default: float32.
        data_preprocessing: A `DataPreprocessing` subclass object to manage
            real-time data pre-processing when training and predicting (such
            as zero center data, std normalization...).
        data_augmentation: `DataAugmentation`. A `DataAugmentation` subclass
            object to manage real-time data augmentation while training (
            such as random image crop, random image flip, random sequence
            reverse...).
        name: `str`. A name for this layer (optional).

    """
    if not shape and not placeholder:
        raise Exception("`shape` or `placeholder` argument is required.")

    if not placeholder:
        # Add 'None' if missing
        assert shape is not None, "A shape or a placeholder must be provided."
        if len(shape) > 1:
            if shape[0] is not None:
                shape = list(shape)
                shape = [None] + shape

        with tf.variable_scope(name) as scope:
            scope = scope.name + '/'
        # with tf.name_scope(name) as scope:
            placeholder = tf.placeholder(shape=shape, dtype=dtype, name="X")

    # Keep track of inputs
    tf.add_to_collection(tf.GraphKeys.INPUTS, placeholder)
    # Keep track of data preprocessing and augmentation
    tf.add_to_collection(tf.GraphKeys.DATA_PREP, data_preprocessing)
    tf.add_to_collection(tf.GraphKeys.DATA_AUG, data_augmentation)

    return placeholder


def fully_connected(incoming, n_units, activation='linear', bias=True,
                    weights_init='truncated_normal', bias_init='zeros',
                    regularizer=None, weight_decay=0.001, trainable=True,
                    restore=True, name="FullyConnected"):
    """ Fully Connected.

    A fully connected layer.

    Input:
        (2+)-D Tensor [samples, input dim]. If not 2D, input will be flatten.

    Output:
        2D Tensor [samples, n_units].

    Arguments:
        incoming: `Tensor`. Incoming (2+)D Tensor.
        n_units: `int`, number of units for this layer.
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
            loading a model.
       name: A name for this layer (optional). Default: 'FullyConnected'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Tensor`. Variable representing units weights.
        b: `Tensor`. Variable representing biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    n_inputs = int(np.prod(input_shape[1:]))

    # Build variables and inference.
    with tf.variable_scope(name) as scope:
        scope = scope.name + '/'

    # with tf.name_scope(name) as scope:
        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None

        # if regularizer:
        #     W_regul = lambda x: losses.get(regularizer)(x, weight_decay)

        W = va.variable(scope + 'W', shape=[n_inputs, n_units],
                        regularizer=W_regul, initializer=W_init,
                        trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, W)

        b = None
        if bias:
            b_init = initializations.get(bias_init)()
            b = va.variable(scope + 'b', shape=[n_units],
                            initializer=b_init, trainable=trainable,
                            restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, b)

        inference = incoming
        # If input is not 2d, flatten it.
        if len(input_shape) > 2:
            inference = tf.reshape(inference, [-1, n_inputs])

        inference = tf.matmul(inference, W)
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


def dropout(incoming, keep_prob, name="Dropout"):
    """ Dropout.

    Outputs the input element scaled up by `1 / keep_prob`. The scaling is so
    that the expected sum is unchanged.

    Arguments:
        incoming : A `Tensor`. The incoming tensor.
        keep_prob : A float representing the probability that each element
            is kept.
        name : A name for this layer (optional).

    References:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
        N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
        (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.

    Links:
      [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf]
        (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

    """

    # with tf.name_scope(name) as scope:
    with tf.variable_scope(name) as scope:
        scope = scope.name + '/'
        
        inference = incoming

        def apply_dropout():
            if type(inference) in [list, np.array]:
                for x in inference:
                    x = tf.nn.dropout(x, keep_prob)
                return inference
            else:
                return tf.nn.dropout(inference, keep_prob)

        is_training = get_training_mode()
        inference = tf.cond(is_training, apply_dropout, lambda: inference)

    return inference


def flatten(incoming, name="Flatten"):
    """ Flatten.

    Flatten the incoming Tensor.

    Input:
        (2+)-D `Tensor`.

    Output:
        2-D `Tensor` [batch, flatten_dims].

    Arguments:
        incoming: `Tensor`. The incoming tensor.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    dims = int(np.prod(input_shape[1:]))
    return reshape(incoming, [-1, dims], name)


def activation(incoming, activation='linear'):

    """ Activation.

    Apply given activation to incoming tensor.

    Arguments:
        incoming: A `Tensor`. The incoming tensor.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.

    """

    if isinstance(activation, str):
        return activations.get(activation)(incoming)
    elif hasattr(incoming, '__call__'):
        return activation(incoming)
    else:
        raise ValueError('Unknown activation type.')

