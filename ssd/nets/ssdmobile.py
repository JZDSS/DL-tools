import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope
import numpy as np

from basenets import mobilenet
from basenets.utils import dropblock
from ssd.nets import ssdbase


class SSDMobileNet(ssdbase.SSDBase, mobilenet.MobileNet):

    def __init__(self,
                 inputs,
                 num_classes,
                 ground_truth,
                 anchor_config,
                 name='SSD_MobileNet',
                 npy_path=None,
                 weight_decay=0.0004,
                 **kwargs):
        super(SSDMobileNet, self).__init__(inputs, name, npy_path, weight_decay=weight_decay,
                                         **kwargs)
        self.ground_truth = ground_truth
        self.num_classes = num_classes
        self.src = anchor_config['src']
        self.aspect_ratios = anchor_config['aspect_ratios']
        self.extra_anchor = anchor_config['extra_anchor']
        self.num_anchors = [len(ratio) + int(extra) for ratio, extra in zip(self.aspect_ratios, self.extra_anchor)]
        self.ext_anchors = anchor_config['extra_scales']
        self.base_net()
        self.extra_net()
        self.predict()
        if npy_path:
            self.ssd_setup()

    def base_net(self):
        del self.endpoints['fc1']
        del self.endpoints['global_pooling']
        self.down_sample()

    def down_sample(self):
        pass

    def extra_net(self):
        endpoints = self.endpoints
        with arg_scope([layers.conv2d],
                       padding='SAME',
                       activation_fn=tf.nn.relu6,
                       weights_regularizer=layers.l2_regularizer(self.weight_decay),
                       normalizer_fn=layers.batch_norm,
                       normalizer_params={'is_training': self.is_training}):
            y = endpoints['Pointwise_Conv2d_13']
            y = layers.conv2d(y, 256, (1, 1), 1, scope='conv1')
            endpoints['conv1'] = y
            y = layers.conv2d(y, 512, (3, 3), 2, scope='conv2')
            # y = dropblock(y, 0.9, 7, self.is_training)
            endpoints['conv2'] = y
            y = layers.conv2d(y, 128, (1, 1), 1, scope='conv3')
            endpoints['conv3'] = y
            y = layers.conv2d(y, 256, (3, 3), 2, scope='conv4')
            # y = dropblock(y, 0.9, 7, self.is_training)
            endpoints['conv4'] = y
            y = layers.conv2d(y, 128, (1, 1), 1, scope='conv5')
            endpoints['conv5'] = y
            y = layers.conv2d(y, 256, (3, 3), 2, scope='conv6')
            # y = dropblock(y, 0.9, 7, self.is_training)
            endpoints['conv6'] = y
            y = layers.conv2d(y, 64, (1, 1), 1, scope='conv7')
            endpoints['conv7'] = y
            y = layers.conv2d(y, 128, (3, 3), 2, scope='conv8')
            # y = dropblock(y, 0.9, 7, self.is_training)
            endpoints['conv8'] = y

    def ssd_setup(self):
        weight_dict = np.load(self.npy_path).item()
        with tf.variable_scope('', reuse=True):
            for k in weight_dict:
                init_op = tf.get_variable(k).assign(weight_dict[k])
                tf.add_to_collection(tf.GraphKeys.INIT_OP, init_op)
