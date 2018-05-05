import tensorflow as tf
import tensorflow.contrib.layers as layers

def atrous_conv2d(inputs,
                  num_outputs,
                  kernel_size,
                  rate=1,
                  padding='SAME',
                  activation_fn=tf.nn.relu,
                  weights_initializer=layers.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=tf.zeros_initializer(),
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None):
    shape = inputs.get_shape().as_list()

    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable('weights',
                            shape=[kernel_size[0], kernel_size[1], shape[-1], num_outputs],
                            dtype=tf.float32,
                            initializer=weights_initializer,
                            regularizer=weights_regularizer,
                            trainable=trainable,
                            collections=variables_collections)
        b = tf.get_variable('biases',
                            shape=[num_outputs],
                            dtype=tf.float32,
                            initializer=biases_initializer,
                            regularizer=biases_regularizer,
                            trainable=trainable,
                            collections=variables_collections)
        y = tf.nn.atrous_conv2d(inputs, w, rate, padding)
        y = tf.nn.bias_add(y, b)
        if activation_fn:
            y = activation_fn(y)
        if outputs_collections:
            tf.add_to_collection(outputs_collections, y)
        return y


def unpool(inputs):
    pass


def safe_division(x, y):
    return tf.case([(tf.equal(y, 0), lambda: tf.zeros_like(x))], lambda: x / y)