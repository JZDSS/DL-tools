import tensorflow as tf
from muti_gpus import layers

def block(inputs, num_outputs, weight_decay, scope, is_training, down_sample = False):
    with tf.variable_scope(scope):
        num_inputs = inputs.get_shape().as_list()[3]

        res = layers.conv(inputs, num_outputs = num_outputs, kernel_size=[3, 3],
                          strides=[1, 2, 2, 1] if down_sample else [1, 1, 1, 1],
                          scope='conv1', b_norm=True, is_training=is_training, weight_decay=weight_decay)

        res = layers.conv(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                          scope='conv2', b_norm=True, is_training=is_training, weight_decay=weight_decay)
        if  num_inputs != num_outputs:
            inputs = layers.conv(inputs, num_outputs=num_outputs, kernel_size=[1, 1], activation_fn=None,
                                 scope='short_cut', strides=[1, 2, 2,1 ], b_norm=True, is_training=is_training,
                                 weight_decay=weight_decay)
        res = tf.nn.relu(res + inputs)

    return res


def build_single_net(x, is_training, FLAGS):
    n = FLAGS.blocks
    # shape = x.get_shape().as_list()
    with tf.variable_scope('pre'):
        pre = layers.conv(x, num_outputs=16,  kernel_size = [3, 3], scope='conv', b_norm=True, is_training=is_training,
                          weight_decay=FLAGS.weight_decay)
        # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
    h = pre
    for i in range(1, n + 1):
        h = block(h, 16, FLAGS.weight_decay, '16_block{}'.format(i), is_training)

    h = block(h, 32, FLAGS.weight_decay, '32_block1', is_training, True)
    for i in range(2, n + 1):
        h = block(h, 32, FLAGS.weight_decay, '32_block{}'.format(i), is_training)

    h = block(h, 64, FLAGS.weight_decay, '64_block1', is_training, True)
    for i in range(2, n + 1):
        h = block(h, 64, FLAGS.weight_decay, '64_block{}'.format(i), is_training)

    shape = h.get_shape().as_list()

    h = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool')
    res = layers.conv(h, num_outputs=FLAGS.num_classes, kernel_size=[1, 1], scope='fc1', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)

    return h, tf.reshape(res, [-1, FLAGS.num_classes])



def build_net(x, is_training, FLAGS):
    confs = []
    ys = []
    with tf.variable_scope('compress70'):
        conf1, y1 = build_single_net(x, False, FLAGS)
        conf1 = tf.sigmoid(layers.conv(conf1, num_outputs=1, kernel_size=[1, 1], scope='fc2', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None))
        confs.append(conf1)
        ys.append(y1)

    with tf.variable_scope('compress90'):
        conf2, y2 = build_single_net(x, False, FLAGS)
        conf2 = tf.sigmoid(layers.conv(conf2, num_outputs=1, kernel_size=[1, 1], scope='fc2', padding='VALID',
                                       b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay,
                                       activation_fn=None))
        confs.append(conf2)
        ys.append(y2)

    with tf.variable_scope('gamma0_8'):
        conf3, y3 = build_single_net(x, False, FLAGS)
        conf3 = tf.sigmoid(layers.conv(conf3, num_outputs=1, kernel_size=[1, 1], scope='fc2', padding='VALID',
                                       b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay,
                                       activation_fn=None))
        confs.append(conf3)
        ys.append(y3)

    with tf.variable_scope('gamma1_2'):
        conf4, y4 = build_single_net(x, False, FLAGS)
        conf4 = tf.sigmoid(layers.conv(conf4, num_outputs=1, kernel_size=[1, 1], scope='fc2', padding='VALID',
                                       b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay,
                                       activation_fn=None))
        confs.append(conf4)
        ys.append(y4)

    with tf.variable_scope('resize0_5'):
        conf5, y5 = build_single_net(x, False, FLAGS)
        conf5 = tf.sigmoid(layers.conv(conf5, num_outputs=1, kernel_size=[1, 1], scope='fc2', padding='VALID',
                                       b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay,
                                       activation_fn=None))
        confs.append(conf5)
        ys.append(y5)

    with tf.variable_scope('resize0_8'):
        conf6, y6 = build_single_net(x, False, FLAGS)
        conf6 = tf.sigmoid(layers.conv(conf6, num_outputs=1, kernel_size=[1, 1], scope='fc2', padding='VALID',
                                       b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay,
                                       activation_fn=None))
        confs.append(conf6)
        ys.append(y6)

    with tf.variable_scope('resize1_5'):
        conf7, y7 = build_single_net(x, False, FLAGS)
        conf7 = tf.sigmoid(layers.conv(conf7, num_outputs=1, kernel_size=[1, 1], scope='fc2', padding='VALID',
                                       b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay,
                                       activation_fn=None))
        confs.append(conf7)
        ys.append(y7)

    with tf.variable_scope('resize2_0'):
        conf8, y8 = build_single_net(x, False, FLAGS)
        conf8 = tf.sigmoid(layers.conv(conf8, num_outputs=1, kernel_size=[1, 1], scope='fc2', padding='VALID',
                                       b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay,
                                       activation_fn=None))
        confs.append(conf8)
        ys.append(y8)

    return confs, ys
