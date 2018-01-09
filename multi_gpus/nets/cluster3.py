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

def build_single_net_w(x, is_training, FLAGS):
    with tf.variable_scope('pre'):
        pre = layers.conv(x, num_outputs=32,  kernel_size = [3, 3], scope='conv', b_norm=True, is_training=is_training,
                          weight_decay=FLAGS.weight_decay)
        # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
    h = pre
    # for i in range(1, n + 1):
    h = block(h, 32, FLAGS.weight_decay, '32_block{}'.format(1), is_training)

    h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
    # h = block(h, 64, FLAGS.weight_decay, '64_block1', is_training, True)
    h = layers.conv(h, num_outputs = 64, kernel_size=[3, 3], strides=[1, 1, 1, 1],
                    scope='conv1', b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay)
    
    # for i in range(2, n + 1):
    h = block(h, 64, FLAGS.weight_decay, '64_block{}'.format(1), is_training)

    h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
    h = layers.conv(h, num_outputs = 128, kernel_size=[3, 3], strides=[1, 1, 1, 1],
                    scope='conv2', b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay)
    
    # for i in range(2, n + 1):
    h = block(h, 128, FLAGS.weight_decay, '128_block{}'.format(1), is_training)

    shape = h.get_shape().as_list()

    h = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool')

    return h

def build_single_net(x, is_training, FLAGS):
    with tf.variable_scope('pre'):
        pre = layers.conv(x, num_outputs=32,  kernel_size = [3, 3], scope='conv', b_norm=True, is_training=is_training,
                          weight_decay=FLAGS.weight_decay)
        # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
    h = pre
    # for i in range(1, n + 1):
    h = block(h, 32, FLAGS.weight_decay, '32_block{}'.format(1), is_training)

    shape = h.get_shape().as_list()
    f1 = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool1')

    h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
    # h = block(h, 64, FLAGS.weight_decay, '64_block1', is_training, True)
    h = layers.conv(h, num_outputs = 64, kernel_size=[3, 3], strides=[1, 1, 1, 1],
                    scope='conv1', b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay)
    
    # for i in range(2, n + 1):
    h = block(h, 64, FLAGS.weight_decay, '64_block{}'.format(1), is_training)

    shape = h.get_shape().as_list()
    f2 = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool2')

    h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
    h = layers.conv(h, num_outputs = 128, kernel_size=[3, 3], strides=[1, 1, 1, 1],
                    scope='conv2', b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay)
    
    # for i in range(2, n + 1):
    h = block(h, 128, FLAGS.weight_decay, '128_block{}'.format(1), is_training)

    shape = h.get_shape().as_list()
    f3 = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool3')

    return tf.concat([f1, f2, f3], axis=3)



def build_net(x, is_training, FLAGS):
    num_branches = FLAGS.num_branches
    y = []
    for i in range(1, num_branches + 1):
        with tf.variable_scope('branch%d' % i):
            tmp = build_single_net(x, is_training, FLAGS)
            if FLAGS.p_relu:
                tmp = layers.p_relu(tmp)
        y.append(tmp)

    with tf.variable_scope('branch%d' % (num_branches + 1)):
        y4 = build_single_net_w(x, is_training, FLAGS)
        
    con = tf.concat(y, axis=3) #192
    con = tf.reshape(con, [-1, 1, num_branches*224])

    w = layers.conv(y4, num_outputs=FLAGS.num_classes*num_branches*224, kernel_size=[1, 1], scope='fc2', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)
    w = tf.reshape(w, [-1, num_branches*224, FLAGS.num_classes])

    res = tf.matmul(con, w)
    res = layers.batch_norm(res, is_training=is_training, scope='bn')
    return tf.reshape(res, [-1, FLAGS.num_classes])