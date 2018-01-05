import tensorflow as tf
from tensorflow.python.training import moving_averages

def _variable_on_cpu(name, shape, initializer, trainable):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape,
                              initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _create_variable(name, shape, initializer, weight_decay=None, trainable=True):

    var = _variable_on_cpu(name, shape, initializer, trainable)
    if weight_decay is not None:
        regularization = tf.multiply(tf.nn.l2_loss(var), weight_decay, name=name + '_regularization')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularization)
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
         scope=None,
         is_training=True,
         trainable=True):
    x_shape = x.get_shape().as_list()

    with tf.variable_scope(scope):

        w = _create_variable('weights', (kernel_size[0], kernel_size[1], x_shape[-1], num_outputs),
                             tf.truncated_normal_initializer(stddev=0.1),
                             weight_decay, trainable)

        y = tf.nn.conv2d(x, w, strides, padding)
        if b_norm:
            y = batch_norm(y, scope='b_norm', is_training=is_training)
        else:
            b = _create_variable('biases', num_outputs, tf.zeros_initializer(), weight_decay, trainable)
            y = tf.nn.bias_add(y, b)
        if activation_fn is not None:
            y = activation_fn(y, name='activation')
    return y


def block(inputx,scope,is_training):
    with tf.variable_scope(scope):
        num_input = inputx.get_shape().as_list()[3]
        bn = conv(inputx, num_input, [3, 3], b_norm=True, scope='conv1', is_training=is_training)
        bn = conv(bn, num_input, [3, 3], b_norm=True, scope='conv2', is_training=is_training, activation_fn=None)
        res = tf.nn.relu(bn+inputx)
    return res

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def build_net(x, is_training, FLAGS):
    # with tf.name_scope('conv1'):
    h_1 = conv(x, 24, [3, 3], b_norm=True, scope='conv1', is_training=is_training)

    # with tf.name_scope('block1'):            
    block1 = block(h_1,scope='block1',is_training=is_training)
    with tf.name_scope('pool1'):          
        pool1 = avg_pool_2x2(block1)
    # with tf.name_scope('conv2'):
    h_2 = conv(pool1, 48, [3, 3], b_norm=True, scope='conv2', is_training=is_training)
    # with tf.name_scope('block2'):            
    block2 = block(h_2,scope='block2',is_training=is_training)
    # with tf.name_scope('conv3'):
    h_3 = conv(block2, 48, [3, 3], b_norm=True, scope='conv3', is_training=is_training)
        # W_3 = weight_variable([3, 3, 48, 48])
        # h_3 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(block2, W_3) ,is_training=phase))
    # with tf.name_scope('block3'):            
    block3 = block(h_3,scope='block3',is_training=is_training)
    with tf.name_scope('pool2'):          
        pool2 = avg_pool_2x2(block3)
    # with tf.name_scope('conv4'):
    h_4 = conv(pool2, 96, [3, 3], b_norm=True, scope='conv4', is_training=is_training)
        # W_4 = weight_variable([3, 3, 48, 96])
        # h_4 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(pool2, W_4) ,is_training=phase))
    # with tf.name_scope('block4'):            
    block4 = block(h_4,scope='block4',is_training=is_training)
    # with tf.name_scope('conv5'):
    h_5 = conv(block4, 96, [3, 3], b_norm=True, scope='conv5', is_training=is_training)
        # W_5 = weight_variable([3, 3, 96, 96])
        # h_5 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(block4, W_5) ,is_training=phase))
    # with tf.name_scope('block5'):            
    block5 = block(h_5,scope='block5',is_training=is_training)
    with tf.name_scope('pool3'):           
        pool3 = avg_pool_2x2(block5)
    # with tf.name_scope('conv6'):
    h_6 = conv(pool3, 96, [3, 3], b_norm=True, scope='conv6', is_training=is_training)
        # W_6 = weight_variable([3, 3, 96, 96])
        # h_6 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(pool3, W_6) ,is_training=phase))
    # with tf.name_scope('block6'):            
    block6 = block(h_6,scope='block6',is_training=is_training)  
    # with tf.name_scope('fc1'):
    y_conv = conv(block6, 10, [8, 8], b_norm=True, scope='fc1', is_training=is_training, padding='VALID', activation_fn=None)
        # W_fc1 = weight_variable([8 * 8 * 96, 10])
        # h_flat = tf.reshape(block6, [-1, 8*8*96])
        # y_conv = tf.contrib.layers.batch_norm(tf.matmul(h_flat, W_fc1),is_training=phase)
    return tf.reshape(y_conv, [-1, 10])