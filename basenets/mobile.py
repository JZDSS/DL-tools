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
        self.build()
        if self.npy_path:
            self.setup()

    def set_npy_path(self, path):
        self.npy_path = path

    def build(self):
        endpoints = self.endpoints
        y = self.inputs['images']
        with arg_scope([layers.conv2d, layers.separable_conv2d], activation_fn=tf.nn.relu6,
                       weights_regularizer=layers.l2_regularizer(self.weight_decay),
                       biases_regularizer=layers.l2_regularizer(self.weight_decay), ):
            y = layers.conv2d(y, 32, (3, 3), 2, 'SAME', scope='Conv2d_0')
            endpoints['Conv2d_0'] = y

            # set num_outputs to None to skipe point-wise convolution
            y = layers.separable_conv2d(y, None, [3, 3], 1, activation_fn=tf.nn.relu6,
                                        normalizer_fn=layers.batch_norm, normalizer_params={'is_train': self.is_train})
            self.outputs['logits'] = tf.squeeze(y)

    def calc_loss(self):
        pass

    def get_update_ops(self):
        return []

    def setup(self):
        return []


if __name__ == '__main__':

    x = tf.placeholder(shape=[None, 227, 227, 3], dtype=tf.float32)
    net = AlexNet(x, npy_path='../npy/alexnet.npy')
    pred = net.outputs['logits']

    init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)

    import cv2
    img = cv2.imread('../images/dog.jpg')
    img = cv2.resize(img, (227, 227))
    img = img
    img = np.expand_dims(img, 0)
    with tf.Session() as sess:
        sess.run(init_ops)
        print(sess.run(tf.argmax(pred, 0), feed_dict={x: img}))

