import tensorflow as tf
from multi_gpus import layers

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


def build_single_net(x, is_training, FLAGS, nn=10):
    n = FLAGS.blocks
    # shape = x.get_shape().as_list()
    with tf.variable_scope('pre'):
        pre = layers.conv(x, num_outputs=24,  kernel_size=[3, 3], scope='conv', b_norm=True, is_training=is_training,
                          weight_decay=FLAGS.weight_decay)
        # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
    h = pre
    for i in range(1, n + 1):
        h = block(h, 24, FLAGS.weight_decay, '24_block{}'.format(i), is_training)

    h = block(h, 48, FLAGS.weight_decay, '46_block1', is_training, True)
    for i in range(2, n + 1):
        h = block(h, 48, FLAGS.weight_decay, '48_block{}'.format(i), is_training)

    h = block(h, 96, FLAGS.weight_decay, '96_block1', is_training, True)
    for i in range(2, n + 1):
        h = block(h, 96, FLAGS.weight_decay, '96_block{}'.format(i), is_training)

    shape = h.get_shape().as_list()

    h = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool')
    res = layers.conv(h, num_outputs=nn, kernel_size=[1, 1], scope='fc1', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)

    return res



def build_net(x, is_training, FLAGS):
    num_branches = FLAGS.num_branches
    y = []
    for i in range(1, num_branches + 1):
        with tf.variable_scope('branch%d' % i):
            tmp = build_single_net(x, is_training, FLAGS)
        y.append(tf.reshape(tmp, [-1, FLAGS.num_classes]))

    with tf.variable_scope('branch%d' % (num_branches + 1)):
        y4 = build_single_net(x, is_training, FLAGS, num_branches)
        y4 = tf.reshape(y4, [-1, num_branches])
    return y4, y
    # con = tf.concat(y, axis=3) #192
    # con = tf.reshape(con, [-1, 1, num_branches*64])

    # w = layers.conv(y4, num_outputs=FLAGS.num_classes*num_branches*64, kernel_size=[1, 1], scope='fc2', padding='VALID',
    #                 b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)
    # w = tf.reshape(w, [-1, num_branches*64, FLAGS.num_classes])

    # res = tf.matmul(con, w)
    # res = layers.batch_norm(res, is_training=is_training, scope='bn')
    # return tf.reshape(res, [-1, FLAGS.num_classes])