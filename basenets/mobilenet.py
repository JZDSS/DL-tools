import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from basenets import net
import numpy as np


class MobileNet(net.Net):

    def __init__(self, inputs, name='AlexNet', npy_path=None, weight_decay=0.00004, **kwargs):
        super(MobileNet, self).__init__(weight_decay=weight_decay, name=name, **kwargs)
        self.inputs = inputs
        self.npy_path = npy_path
        # self.is_training = tf.constant(True, dtype=tf.bool)
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[])
        self.build()
        if self.npy_path:
            self.setup()

    def set_npy_path(self, path):
        self.npy_path = path

    def build(self):
        endpoints = self.endpoints
        y = self.inputs['images']
        with arg_scope([layers.conv2d, layers.separable_conv2d],
                       padding='SAME',
                       activation_fn=tf.nn.relu6,
                       weights_initializer=layers.xavier_initializer(),
                       weights_regularizer=layers.l2_regularizer(self.weight_decay),
                       normalizer_fn=layers.batch_norm,
                       normalizer_params={'is_training': self.is_training}):
            y = layers.conv2d(y, 32, (3, 3), 2, scope='Conv2d_0')
            endpoints['Conv2d_0'] = y

            # set num_outputs to None to skipe point-wise convolution
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=1, scope='Depthwise_Conv2d_1')
            y = layers.conv2d(y, 64, (1, 1), scope='Pointwise_Conv2d_1')
            endpoints['Pointwise_Conv2d_1'] = y
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=2, scope='Depthwise_Conv2d_2')
            y = layers.conv2d(y, 128, (1, 1), scope='Pointwise_Conv2d_2')
            endpoints['Pointwise_Conv2d_2'] = y
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=1, scope='Depthwise_Conv2d_3')
            y = layers.conv2d(y, 128, (1, 1), scope='Pointwise_Conv2d_3')
            endpoints['Pointwise_Conv2d_3'] = y
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=2, scope='Depthwise_Conv2d_4')
            y = layers.conv2d(y, 256, (1, 1), scope='Pointwise_Conv2d_4')
            endpoints['Pointwise_Conv2d_4'] = y
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=1, scope='Depthwise_Conv2d_5')
            y = layers.conv2d(y, 256, (1, 1), scope='Pointwise_Conv2d_5')
            endpoints['Pointwise_Conv2d_5'] = y
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=2, scope='Depthwise_Conv2d_6')
            y = layers.conv2d(y, 512, (1, 1), scope='Pointwise_Conv2d_6')
            endpoints['Pointwise_Conv2d_6'] = y
            # repeat 5 times
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=1, scope='Depthwise_Conv2d_7')
            y = layers.conv2d(y, 512, (1, 1), scope='Pointwise_Conv2d_7')
            endpoints['Pointwise_Conv2d_7'] = y
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=1, scope='Depthwise_Conv2d_8')
            y = layers.conv2d(y, 512, (1, 1), scope='Pointwise_Conv2d_8')
            endpoints['Pointwise_Conv2d_8'] = y
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=1, scope='Depthwise_Conv2d_9')
            y = layers.conv2d(y, 512, (1, 1), scope='Pointwise_Conv2d_9')
            endpoints['Pointwise_Conv2d_9'] = y
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=1, scope='Depthwise_Conv2d_10')
            y = layers.conv2d(y, 512, (1, 1), scope='Pointwise_Conv2d_10')
            endpoints['Pointwise_Conv2d_10'] = y
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=1, scope='Depthwise_Conv2d_11')
            y = layers.conv2d(y, 512, (1, 1), scope='Pointwise_Conv2d_11')
            endpoints['Pointwise_Conv2d_11'] = y
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=2, scope='Depthwise_Conv2d_12')
            y = layers.conv2d(y, 1024, (1, 1), scope='Pointwise_Conv2d_12')
            endpoints['Pointwise_Conv2d_12'] = y
            # 此层stride存疑，原文为2。
            y = layers.separable_conv2d(y, None, (3, 3), 1, stride=1, scope='Depthwise_Conv2d_13')
            y = layers.conv2d(y, 1024, (1, 1), scope='Pointwise_Conv2d_13')
            endpoints['Pointwise_Conv2d_13'] = y
            y = layers.avg_pool2d(y, (7, 7), 1, 'VALID', scope='global_pooling')
            endpoints['global_pooling'] = y
            y = layers.flatten(y)
            y = layers.fully_connected(y, 1000, scope='fc1')
            endpoints['fc1'] = y
            self.outputs['logits'] = y
        return y

    def calc_loss(self):
        pass

    def get_update_ops(self):
        return []

    def setup(self):
        return []


if __name__ == '__main__':

    x = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
    net = MobileNet({'images': x}, npy_path='../npy/alexnet.npy')
    pred = net.outputs['logits']

    init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)

    import cv2
    img = cv2.imread('../images/dog.jpg')
    img = cv2.resize(img, (224, 224))
    img = img
    img = np.expand_dims(img, 0)
    with tf.Session() as sess:
        sess.run(init_ops)
        print(sess.run(tf.argmax(pred, 0), feed_dict={x: img}))

