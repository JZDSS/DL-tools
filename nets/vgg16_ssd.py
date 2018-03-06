import tensorflow as tf
from tensorflow.contrib import layers
from nets import net, ssd
import numpy as np

class vgg16_ssd(net.Net):

    def __init__(self, vgg16_path='../ssd/vgg16/vgg16.npy'):
        super(vgg16_ssd, self).__init__(name='vgg16_ssd')
        self.vgg16_path = vgg16_path

    def set_pre_trained_weight_path(self, path):
        self.vgg16_path = path

    def build(self, inputs):
        # y = layers.conv2d(inputs, 64, [3, 3], 1, 'SAME', scope='conv1_1')
        # y = layers.conv2d(y, 64, [3, 3], 1, 'SAME', scope='conv1_2')
        y = inputs
        y = layers.repeat(y, 2, layers.conv2d, 64, [3, 3], 1, 'SAME', scope='conv1')
        y = layers.max_pool2d(y, [2, 2], 2, 'SAME', scope='pool1')
        y = layers.repeat(y, 2, layers.conv2d, 128, [3, 3], 1, 'SAME', scope='conv2')
        y = layers.max_pool2d(y, [2, 2], 2, 'SAME', scope='pool2')
        y = layers.repeat(y, 3, layers.conv2d, 256, [3, 3], 1, 'SAME', scope='conv3')
        y = layers.max_pool2d(y, [2, 2], 2, 'SAME', scope='pool3')
        y = layers.repeat(y, 3, layers.conv2d, 512, [3, 3], 1, 'SAME', scope='conv4')
        y = layers.max_pool2d(y, [2, 2], 2, 'SAME', scope='pool4')
        y = layers.repeat(y, 3, layers.conv2d, 512, [3, 3], 1, 'SAME', scope='conv5')
        # y = layers.max_pool2d(y, [2, 2], 2, 'SAME', scope='pool5')
        # change pool5 from 2 × 2 − s2 to 3 × 3 − s1
        y = layers.max_pool2d(y, [3, 3], 1, 'SAME', scope='pool5')
        # y = layers.fully_connected(layers.flatten(y), 4096, scope='fc6')
        # y = layers.fully_connected(layers.flatten(y), 4096, scope='fc7')
        # y = layers.conv2d(y, 4096, [7, 7], 1, 'VALID', scope='fc6')
        # y = layers.conv2d(y, 4096, [1, 1], 1, 'VALID', scope='fc7')
        with tf.variable_scope('fc6'):
            w = tf.get_variable('weights', shape=[3, 3, 512, 1024], dtype=tf.float32)
            b = tf.get_variable('biases', shape=[1024], dtype=tf.float32)
            y = tf.nn.atrous_conv2d(y, w, 6, 'SAME')
            y = tf.nn.bias_add(y, b)
        y = layers.conv2d(y, 1024, [1, 1], 1, 'VALID', scope='fc7')

        # y = layers.fully_connected(layers.flatten(y), 1000, scope='fc8')
        return y






    def get_update_ops(self):
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def loss(self, logits, labels, *args, **kwargs):
        pass

    def setup(self):
        """Define ops that load pre-trained vgg16 net's weights and biases and add them to tf.GraphKeys.INIT_OP
        collection.
        """

        # caffe-tensorflow/convert.py can only run with Python2. Since the default encoding format of Python2 is ASCII
        # but the default encoding format of Python3 is UTF-8, it will raise an error without 'encoding="latin1"'
        weight_dict = np.load(self.vgg16_path, encoding="latin1").item()

        scopes = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                  'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
        for scope in scopes:
            with tf.variable_scope(scope.split('_')[0] + '/' + scope, reuse=True):
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')
                w_init_op = weights.assign(weight_dict[scope]['weights'])
                b_init_op = biases.assign(weight_dict[scope]['biases'])
                tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)

        with tf.variable_scope('fc6', reuse=True):
            weights = tf.get_variable('weights')
            biases = tf.get_variable('biases')
            w = weight_dict['fc6']['weights']
            b = weight_dict['fc6']['biases']
            w = np.reshape(w, (7, 7, 512, 4096))
            w = w[0:-1:2, 0:-1:2, :, 0:-1:4]
            b = b[0:-1:4]
            w_init_op = weights.assign(w)
            b_init_op = biases.assign(b)
            tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
            tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)

        with tf.variable_scope('fc7', reuse=True):
            weights = tf.get_variable('weights')
            biases = tf.get_variable('biases')
            w = weight_dict['fc7']['weights']
            b = weight_dict['fc7']['biases']
            w = np.reshape(w, (1, 1, 4096, 4096))
            w = w[:, :, 0:-1:4, 0:-1:4]
            b = b[0:-1:4]
            w_init_op = weights.assign(w)
            b_init_op = biases.assign(b)
            tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
            tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)


x = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
net = vgg16_ssd()
net.build(x)
net.setup()

init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(init_ops)


