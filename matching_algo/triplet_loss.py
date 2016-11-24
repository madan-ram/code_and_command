import tensorflow as tf


def triplet_loss(Xa, Xp, Xn, margin = 1.0):
    similar_l2_norm = tf.sqrt(tf.reduce_sum(tf.mul(Xa - Xp, Xa - Xp), 1))
    different_l2_norm = tf.sqrt(tf.reduce_sum(tf.mul(Xa - Xn, Xa - Xn), 1))
    s = tf.maximum(0.0, similar_l2_norm -  different_l2_norm + margin)
    s = tf.reduce_mean(s, name='triplet_loss')
    return s, similar_l2_norm, different_l2_norm

