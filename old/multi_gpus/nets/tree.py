from multi_gpus import layers

def merge(w, x, y):
    if FLAGS.p_relu:
        x = layers.p_relu(x)
        y = layers.p_relu(y)

    con = tf.concat(x, y, 3)
    res = tf.multiply(w, con)
    return res
    


def build_net(x, is_training, FLAGS):
    n = FLAGS.blocks
    with tf.variable_scope('pre'):
        pre = layers.conv(x, num_outputs=16,  kernel_size = [3, 3], scope='conv', b_norm=True, is_training=is_training,
                          weight_decay=FLAGS.weight_decay)

    h1 = 