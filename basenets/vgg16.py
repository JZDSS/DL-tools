import tensorflow as tf
from tensorflow.contrib import layers
from basenets import net, ssd
import numpy as np

class vgg16(net.Net):

    def __init__(self, vgg16_path='../ssd/vgg16/vgg16.npy'):
        super(vgg16, self).__init__(name='vgg16_ssd')
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
        y = layers.max_pool2d(y, [2, 2], 2, 'SAME', scope='pool5')
        y = layers.fully_connected(layers.flatten(y), 4096, scope='fc6')
        y = layers.fully_connected(layers.flatten(y), 4096, scope='fc7')
        y = layers.fully_connected(layers.flatten(y), 1000, scope='fc8')
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
        scopes = ['conv1/conv1_1', 'conv1/conv1_2', 'conv2/conv2_1', 'conv2/conv2_2', 'conv3/conv3_1', 'conv3/conv3_2',
                  'conv3/conv3_3', 'conv4/conv4_1', 'conv4/conv4_2', 'conv4/conv4_3', 'conv5/conv5_1', 'conv5/conv5_2',
                  'conv5/conv5_3', 'fc6', 'fc7', 'fc8']
        for scope in scopes:
            with tf.variable_scope(scope, reuse=True):
                weights = tf.get_variable('weights')
                biases = tf.get_variable('biases')
                w_init_op = weights.assign(weight_dict[scope.split('/')[-1]]['weights'])
                b_init_op = biases.assign(weight_dict[scope.split('/')[-1]]['biases'])
                tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)

with tf.device('/cpu:0'):
    x = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
    net = vgg16()
    pred = net.build(x)

    net.setup()

    init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)

    import cv2
    img = cv2.imread('/home/yqi/Pictures/lion.jpeg')
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, 0)
    with tf.Session() as sess:
        # tf.global_variables_initializer().run()
        sess.run(init_ops)
        # with tf.variable_scope('conv1/conv1_1', reuse=True):
        #     print(sess.run(tf.get_variable('weights')))
        print(sess.run(tf.argmax(pred, 1), feed_dict={x: img}))

