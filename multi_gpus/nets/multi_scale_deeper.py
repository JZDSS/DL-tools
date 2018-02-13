from multi_gpus import layers
import tensorflow as tf


def block(inputs, num_outputs, weight_decay, scope, is_training, down_sample = False):
    with tf.variable_scope(scope):
        num_inputs = inputs.get_shape().as_list()[3]

        res = layers.conv(inputs, num_outputs = num_outputs, kernel_size=[3, 3], activation_fn=tf.nn.elu,
                          strides=[1, 2, 2, 1] if down_sample else [1, 1, 1, 1],
                          scope='conv1', b_norm=True, is_training=is_training, weight_decay=weight_decay)

        res = layers.conv(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                          scope='conv2', b_norm=True, is_training=is_training, weight_decay=weight_decay)
        if  num_inputs != num_outputs:
            inputs = layers.conv(inputs, num_outputs=num_outputs, kernel_size=[1, 1], activation_fn=None,
                                 scope='short_cut', strides=[1, 2, 2,1 ], b_norm=True, is_training=is_training,
                                 weight_decay=weight_decay)
        res = tf.nn.elu(res + inputs)

    return res


def build_net(x, is_training, FLAGS):
    n = FLAGS.blocks
    # shape = x.get_shape().as_list()
    with tf.variable_scope('pre'):
        pre = layers.conv(x, num_outputs=64,  kernel_size = [3, 3], scope='conv', b_norm=True, is_training=is_training,
                          weight_decay=FLAGS.weight_decay, activation_fn=tf.nn.elu)
        # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
    h = pre

    shape = h.get_shape().as_list()

    pool0 = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool0')
    fc0 = layers.conv(pool0, num_outputs=FLAGS.num_classes, kernel_size=[1, 1], scope='fc0', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)

    for i in range(1, 4):
        h = block(h, 64, FLAGS.weight_decay, '64_block{}'.format(i), is_training)

    shape = h.get_shape().as_list()

    pool1 = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool1')
    fc1 = layers.conv(pool1, num_outputs=FLAGS.num_classes, kernel_size=[1, 1], scope='fc1', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)

    h = block(h, 128, FLAGS.weight_decay, '128_block1', is_training, True)
    for i in range(2, 4):
        h = block(h, 128, FLAGS.weight_decay, '128_block{}'.format(i), is_training)
    
    shape = h.get_shape().as_list()

    pool2 = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool2')
    fc2 = layers.conv(pool2, num_outputs=FLAGS.num_classes, kernel_size=[1, 1], scope='fc2', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)

    h = block(h, 256, FLAGS.weight_decay, '256_block1', is_training, True)
    for i in range(2, 6):
        h = block(h, 256, FLAGS.weight_decay, '256_block{}'.format(i), is_training)
    
    shape = h.get_shape().as_list()

    pool3 = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool3')
    
    fc3 = layers.conv(pool3, num_outputs=FLAGS.num_classes, kernel_size=[1, 1], scope='fc3', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)

    h = fc0 + fc1 + fc2 + fc3
    return tf.reshape(h, [-1, FLAGS.num_classes])


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_integer('num_classes', 10, '')
    flags.DEFINE_float('weight_decay', 0.00004, '')
    FLAGS = flags.FLAGS
    x = tf.placeholder(tf.float32, (10, 10, 10, 3))
    build_net(x, 1, True, FLAGS)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        writer.flush()

    writer.close()

