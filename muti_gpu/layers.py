from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from tensorflow.python.training import moving_averages
import math


def _variable_on_cpu(name, shape, initializer, regularizer, trainable):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape,
                              initializer=initializer, dtype=dtype,
                              regularizer=regularizer, trainable=trainable)
    return var


def _create_variable(name, shape, initializer, weight_decay=None, trainable=True):

    if weight_decay is not None:
        # regularization = tf.multiply(tf.nn.l2_loss(var), weight_decay, name=name + '_regularization')
        # tf.add_to_collection('losses', regularization)
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    var = _variable_on_cpu(name, shape, initializer, regularizer, trainable)
    return var


def batch_norm(x,
               decay=0.999,
               epsilon=0.001,
               is_training=True,
               scope=None):
    with tf.variable_scope(scope):
        is_training = tf.convert_to_tensor(is_training, dtype=tf.bool, name='is_training')
        x_shape = x.get_shape().as_list()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))

        beta = _create_variable('beta', params_shape,
                                initializer=tf.zeros_initializer)
        gamma = _create_variable('gamma', params_shape,
                                 initializer=tf.ones_initializer)

        moving_mean = _create_variable('moving_mean', params_shape,
                                       initializer=tf.zeros_initializer,
                                       trainable=False)
        moving_variance = _create_variable('moving_variance',
                                           params_shape,
                                           initializer=tf.ones_initializer,
                                           trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, decay, name='update_moving_mean')
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, decay, name='update_moving_variance')
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

        mean, variance = tf.cond(
            is_training, lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)

    return y


def conv(x,
         num_outputs,
         kernel_size,
         strides=[1, 1, 1, 1],
         padding='SAME',
         weight_decay=0.00004,
         b_norm=False,
         activation_fn=tf.nn.relu,
         scope=None):
    x_shape = x.get_shape().as_list()

    with tf.variable_scope(scope):

        w = _create_variable('weights', (kernel_size[0], kernel_size[1], x_shape[-1], num_outputs),
                             tf.truncated_normal_initializer(stddev=math.sqrt(2/(kernel_size[0]*kernel_size[1]*x_shape[-1]))),
                             weight_decay)

        y = tf.nn.conv2d(x, w, strides, padding)
        if b_norm:
            y = batch_norm(y, scope='b_norm')
        else:
            b = _create_variable('biases', num_outputs, tf.zeros_initializer(), weight_decay)
            y = tf.nn.bias_add(y, b)
        if activation_fn is not None:
            y = activation_fn(y, name='relu')
    return y


if __name__ == '__main__':
    x = tf.zeros(dtype=tf.float32, shape=(10,10,10,3))
    y = batch_norm(x, scope='batch_norm')
    y = conv(y, 10, [3, 3], scope='conv', b_norm=True, activation_fn=None)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        writer.flush()

    writer.close()
