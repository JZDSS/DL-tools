import tensorflow as tf
from muti_gpus import layers

def block(inputs, num_outputs, weight_decay, scope, is_training, down_sample = False, trainable=True):
    with tf.variable_scope(scope):
        num_inputs = inputs.get_shape().as_list()[3]

        res = layers.conv(inputs, num_outputs = num_outputs, kernel_size=[3, 3],
                          strides=[1, 2, 2, 1] if down_sample else [1, 1, 1, 1],
                          scope='conv1', b_norm=True, is_training=is_training, weight_decay=weight_decay, trainable=trainable)

        res = layers.conv(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                          scope='conv2', b_norm=True, is_training=is_training, weight_decay=weight_decay, trainable=trainable)
        if  num_inputs != num_outputs:
            inputs = layers.conv(inputs, num_outputs=num_outputs, kernel_size=[1, 1], activation_fn=None,
                                 scope='short_cut', strides=[1, 2, 2,1 ], b_norm=True, is_training=is_training,
                                 weight_decay=weight_decay, trainable=trainable)
        res = tf.nn.relu(res + inputs)

    return res


def build_single_net(x, is_training, FLAGS, trainable=True):
    n = FLAGS.blocks
    # shape = x.get_shape().as_list()
    with tf.variable_scope('pre'):
        pre = layers.conv(x, num_outputs=16,  kernel_size = [3, 3], scope='conv', b_norm=True, is_training=is_training,
                          weight_decay=FLAGS.weight_decay, trainable=trainable)
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
    # res = layers.conv(h, num_outputs=FLAGS.num_classes, kernel_size=[1, 1], scope='fc1', padding='VALID',
    #                 b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)

    return h



def build_net(x, is_training, FLAGS):

    with tf.variable_scope('compress70'):
        y1 = build_single_net(x, is_training, FLAGS, False)

    with tf.variable_scope('compress90'):
        y2 = build_single_net(x, is_training, FLAGS, False)

    with tf.variable_scope('gamma0_8'):
        y3 = build_single_net(x, is_training, FLAGS, False)

    with tf.variable_scope('resize0_5'):
        y4 = build_single_net(x, is_training, FLAGS, False)

    with tf.variable_scope('resize0_8'):
        y5 = build_single_net(x, is_training, FLAGS, False)

    with tf.variable_scope('resize1_5'):
        y6 = build_single_net(x, is_training, FLAGS, False)

    with tf.variable_scope('resize2_0'):
        y7 = build_single_net(x, is_training, FLAGS, False)

    with tf.variable_scope('cluster'):
        y8 = build_single_net(x, is_training, FLAGS)
        
    con = tf.concat([y1, y2, y3, y4, y5, y6, y7], axis=3) #192
    con = tf.reshape(con, [-1, 1, 448])

    w = layers.conv(y8, num_outputs=FLAGS.num_classes*448, kernel_size=[1, 1], scope='fc2', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)
    w = tf.reshape(w, [-1, 448, FLAGS.num_classes])

    res = tf.matmul(con, w)
    res = layers.batch_norm(res, is_training=is_training, scope='bn')
    return tf.reshape(res, [-1, FLAGS.num_classes])