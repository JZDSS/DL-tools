import tensorflow as tf
from tensorflow.contrib import layers
from nets import net, ssd
import numpy as np

class alex(net.Net):

    def __init__(self, alex_path='/home/yqi/Desktop/workspace/SPCup/tfmodels/alexnet.npy'):
        super(alex, self).__init__(name='vgg16_ssd')
        self.alex_path = alex_path



    def set_pre_trained_weight_path(self, path):
        self.vgg16_path = path

    def build(self, inputs):
        # y = layers.conv2d(inputs, 64, [3, 3], 1, 'SAME', scope='conv1_1')
        # y = layers.conv2d(y, 64, [3, 3], 1, 'SAME', scope='conv1_2')
        y = inputs
        y = layers.conv2d(y, 96, [11, 11], 4, 'VALID', scope='conv1')
        y = tf.nn.lrn(y, 5, 1, 0.0001, 0.75)
        y = layers.max_pool2d(y, [3, 3], 2, 'VALID', scope='pool1')
        y1, y2 = tf.split(y, 2, 3)
        y1 = layers.conv2d(y1, 128, [5, 5], 1, 'SAME', scope='conv2_1')
        y2 = layers.conv2d(y2, 128, [5, 5], 1, 'SAME', scope='conv2_2')
        y = tf.concat([y1, y2], 3)
        y = tf.nn.lrn(y, 5, 1, 0.0001, 0.75)
        y = layers.max_pool2d(y, [3, 3], 2, 'VALID', scope='pool2')
        y = layers.conv2d(y, 384, [3, 3], 1, 'SAME', scope='conv3')
        y1, y2 = tf.split(y, 2, 3)
        y1 = layers.conv2d(y1, 192, [3, 3], 1, 'SAME', scope='conv4_1')
        y2 = layers.conv2d(y2, 192, [3, 3], 1, 'SAME', scope='conv4_2')
        y1 = layers.conv2d(y1, 128, [3, 3], 1, 'SAME', scope='conv5_1')
        y2 = layers.conv2d(y2, 128, [3, 3], 1, 'SAME', scope='conv5_2')
        y = tf.concat([y1, y2], 3)
        y = layers.max_pool2d(y, [3, 3], 2, 'VALID', scope='pool5')
        y = layers.fully_connected(layers.flatten(y), 4096, scope='fc6')
        y = layers.fully_connected(layers.flatten(y), 4096, scope='fc7')
        y = layers.fully_connected(layers.flatten(y), 1000, scope='fc8', activation_fn=None)
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
        weight_dict = np.load(self.alex_path, encoding="latin1").item()
        scopes = ['conv1', 'conv3', 'fc6', 'fc7', 'fc8']
        for scope in scopes:
            with tf.variable_scope(scope, reuse=True):
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')
                w_init_op = weights.assign(weight_dict[scope]['weights'])
                b_init_op = biases.assign(weight_dict[scope]['biases'])
                tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)

        scopes = ['conv2', 'conv4', 'conv5']
        for scope in scopes:
            w = weight_dict[scope]['weights']
            w1, w2 = np.split(w, 2, 3)
            b = weight_dict[scope]['biases']
            b1, b2 = np.split(b, 2, 0)
            with tf.variable_scope(scope + '_1', reuse=True):
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')
                w_init_op = weights.assign(w1)
                b_init_op = biases.assign(b1)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)
            with tf.variable_scope(scope + '_2', reuse=True):
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')
                w_init_op = weights.assign(w2)
                b_init_op = biases.assign(b2)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)

x = tf.placeholder(shape=[None, 227, 227, 3], dtype=tf.float32)
net = alex()
pred = net.build(x)

net.setup()

init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)

import cv2
img = cv2.imread('/home/yqi/Pictures/dog.jpg')
img = cv2.resize(img, (227, 227))
img = img
img = np.expand_dims(img, 0)
with tf.Session() as sess:
    # tf.global_variables_initializer().run()
    sess.run(init_ops)
    # with tf.variable_scope('conv1/conv1_1', reuse=True):
    #     print(sess.run(tf.get_variable('weights')))
    print(sess.run(tf.argmax(pred, 1), feed_dict={x: img}))

