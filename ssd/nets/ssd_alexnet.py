import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from nets import alexnet
from nets import extra_layers

class SSD_AlexNet(alexnet.AlexNet):

    def __init__(self, image, name='SSD_AlexNet', npy_path=None):
        super(SSD_AlexNet, self).__init__(image, name, npy_path)
        self.base_net()
        self.extra_net()
        self.predict()
        if npy_path:
            self.ssd_setup()

    def base_net(self):
        del self.endpoints['fc6']
        del self.endpoints['fc7']
        del self.endpoints['fc8']
        self.down_sample()

    def down_sample(self):
        endpoints = self.endpoints
        fc5_1 = self.endpoints['conv5_1']
        fc5_2 = self.endpoints['conv5_2']
        with tf.variable_scope('rebuild'):
            fc5 = tf.concat([fc5_1 ,fc5_2], 3)
            # y = layers.max_pool2d(fc5, [3, 3], 2, 'VALID', scope='pool5')
            y = extra_layers.atrous_conv2d(fc5, 1024, [3, 3], 6, 'VALID', scope='fc6')
            endpoints['fc6'] = y
            y = layers.conv2d(y, 1024, [1, 1], 1, 'VALID', scope='fc7')
            endpoints['fc7'] = y

    def ssd_setup(self):
        weight_dict = np.load(self.npy_path, encoding="latin1").item()
        with tf.variable_scope('rebuild/fc6', reuse=True):
            weights = tf.get_variable('weights')
            biases = tf.get_variable('biases')
            w = weight_dict['fc6']['weights']
            b = weight_dict['fc6']['biases']
            w = np.reshape(w, (6, 6, 256, 4096))
            w = w[0:-1:2, 0:-1:2, :, 0:-1:4]
            b = b[0:-1:4]
            w_init_op = weights.assign(w)
            b_init_op = biases.assign(b)
            tf.add_to_collection(tf.GraphKeys.INIT_OP, w_init_op)
            tf.add_to_collection(tf.GraphKeys.INIT_OP, b_init_op)

        with tf.variable_scope('rebuild/fc7', reuse=True):
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

    def extra_net(self):
        endpoints = self.endpoints
        y = endpoints['fc7']

    def predict(self):
        pass


if __name__ == '__main__':
    x = tf.placeholder(shape=(1, 300, 300, 3), dtype=tf.float32)
    net = SSD_AlexNet(x, npy_path='/home/yqi/Desktop/workspace/PycharmProjects/DL-tools/npy/alexnet.npy')