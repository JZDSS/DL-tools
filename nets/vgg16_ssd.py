from __future__ import division
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from nets import net, ssd
import numpy as np

# s_k = s_min + (s_max - s_min)/(m - 1)(k - 1)
default_anchor_scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]        # of original image size
# s_k^{'} = sqrt(s_k*s_k+1)
default_ext_anchor_scales = [0.26, 0.28, 0.43, 0.60, 0.78, 0.98]  # of original image size
# omitted aspect ratio 1
default_aspect_ratios = [[1/2, 2],                          # conv4_3
                         [1/3, 1/2, 2, 3],                  # conv7
                         [1/3, 1/2, 2, 3],                  # conv8_2
                         [1/3, 1/2, 2, 3],                  # conv9_2
                         [1 / 2, 2],                        # conv10_2
                         [1 / 2, 2]]                        # conv11_2

class vgg16_ssd(net.Net):

    def __init__(self, vgg16_path='ssd/vgg16/vgg16.npy', weight_decay=0.0005,
                 num_classes=20, anchor_scales=default_anchor_scales, aspect_ratios=default_aspect_ratios,
                 ext_anchors=default_ext_anchor_scales):
        super(vgg16_ssd, self).__init__(name='vgg16_ssd')
        self.vgg16_path = vgg16_path
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.anchor_scales = anchor_scales
        # self.num_anchors = len(anchor_scales)
        self.aspect_ratios = aspect_ratios
        self.num_anchors = [len(ratio) + 2 for ratio in self.aspect_ratios]
        self.ext_anchors = ext_anchors
        self.feature_map_size = []
        self.feature_maps = None

    def set_pre_trained_weight_path(self, path):
        self.vgg16_path = path

    def l2_norm(self, x):
        x = tf.nn.l2_normalize(x, [0, 1, 2, 3])
        gamma = tf.get_variable('gamma', shape=[1, 1, 512], initializer=tf.constant_initializer(20, tf.float32))
        x = x * gamma
        return x

    def build(self, inputs):
        feature_maps = []
        with arg_scope([layers.conv2d], weights_initializer=layers.xavier_initializer(),
                       weights_regularizer=layers.l2_regularizer(self.weight_decay), padding='SAME'):
            y = inputs
            y = layers.repeat(y, 2, layers.conv2d, 64, [3, 3], 1, scope='conv1')
            y = layers.max_pool2d(y, [2, 2], 2, 'SAME', scope='pool1')
            y = layers.repeat(y, 2, layers.conv2d, 128, [3, 3], 1, scope='conv2')
            y = layers.max_pool2d(y, [2, 2], 2, 'SAME', scope='pool2')
            y = layers.repeat(y, 3, layers.conv2d, 256, [3, 3], 1, scope='conv3')
            y = layers.max_pool2d(y, [2, 2], 2, 'SAME', scope='pool3')
            y = layers.repeat(y, 3, layers.conv2d, 512, [3, 3], 1, scope='conv4')
            y = self.l2_norm(y)
            feature_maps.append(y)
            y = layers.max_pool2d(y, [2, 2], 2, 'SAME', scope='pool4')
            y = layers.repeat(y, 3, layers.conv2d, 512, [3, 3], 1, scope='conv5')
            y = layers.max_pool2d(y, [3, 3], 1, 'SAME', scope='pool5')
            with tf.variable_scope('fc6'):
                w = tf.get_variable('weights', shape=[3, 3, 512, 1024], dtype=tf.float32)
                b = tf.get_variable('biases', shape=[1024], dtype=tf.float32)
                y = tf.nn.atrous_conv2d(y, w, 6, 'SAME')
                y = tf.nn.bias_add(y, b)
            y = layers.conv2d(y, 1024, [1, 1], 1, scope='fc7')
            feature_maps.append(y)
            y = layers.conv2d(y, 256, [1, 1], 1, scope='conv8_1')
            y = layers.conv2d(y, 512, [3, 3], 2, scope='conv8_2')
            feature_maps.append(y)
            y = layers.conv2d(y, 128, [1, 1], 1, scope='conv9_1')
            y = layers.conv2d(y, 256, [3, 3], 2, scope='conv9_2')
            feature_maps.append(y)
            y = layers.conv2d(y, 128, [1, 1], 1, scope='conv10_1')
            y = layers.conv2d(y, 256, [3, 3], 1, padding='VALID', scope='conv10_2')
            feature_maps.append(y)
            y = layers.conv2d(y, 128, [1, 1], 1, scope='conv11_1')
            y = layers.conv2d(y, 256, [3, 3], 1, padding='VALID',scope='conv11_2')
            feature_maps.append(y)
            self.feature_map_size = [map.get_shape().as_list()[1:3] for map in feature_maps]

            # predictions = []
            self.location = []
            self.classification = []
            for i, feature_map in enumerate(feature_maps):
                num_outputs = self.num_anchors[i] * (self.num_classes + 1 + 4)
                prediction = layers.conv2d(feature_map, num_outputs, [3, 3], 1, scope='pred_%d' % i)

                locations, classifications = tf.split(prediction,
                                                      [self.num_anchors[i] * 4,
                                                       self.num_anchors[i] * (self.num_classes + 1)],
                                                      -1)
                shape = locations.get_shape()
                locations = tf.reshape(locations, [-1,
                                                   shape[1],
                                                   shape[2],
                                                   self.num_anchors[i],
                                                   4])
                shape = classifications.get_shape()
                classifications = tf.reshape(classifications,
                                             [-1,
                                              shape[1],
                                              shape[2],
                                              self.num_anchors[i],
                                              (self.num_classes + 1)])
                self.location.append(locations)
                self.classification.append(classifications)
        self._setup()
        self.feature_maps = feature_maps
        # return predictions

    def get_update_ops(self):
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def loss(self, logits, labels, *args, **kwargs):
        pass

    def _setup(self):
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

    def add_summary(self):
        for i, feature_map in enumerate(self.feature_maps):
            tf.summary.histogram('feature_map_%d' % i,  feature_map)

    def hard_negtave_mining(self, labels):
        """
        :param predictions: List of predictions of all feature maps
        :param labels: List of labels of all fearure maps
        :return:
        """
        predictions = self.classification
        pos_mask_list = []
        neg_mask_list = []
        for i in range(len(predictions)):
            # for every feature map
            prediction = predictions[i]
            label = labels[i]
            pos_mask = tf.not_equal(label, 0)
            neg_mask = tf.logical_not(pos_mask)
            num_positive = tf.reduce_sum(tf.cast(pos_mask, tf.int32))
            num_negtave = tf.minimum(num_positive * 3 + 1, tf.reduce_sum(tf.cast(neg_mask, tf.int32)))
            scores = tf.nn.softmax(prediction)[:, :, :, :, 0]
            neg_scores = tf.cast(neg_mask, tf.float32) * scores
            flat_neg = tf.reshape(neg_scores, [-1])
            val, _ = tf.nn.top_k(flat_neg, num_negtave)
            minval = val[-1]
            neg_mask = tf.greater(flat_neg, minval)
            neg_mask = tf.reshape(neg_mask, pos_mask.shape)
            neg_mask_list.append(neg_mask)
            pos_mask_list.append(pos_mask)

        return pos_mask_list, neg_mask_list

    def smooth_l1_loss(self, x):
        """
        Computes smooth l1 loss: x^2 / 2 if abs(x) < 1, abs(x) - 0.5 otherwise.
        See [Fast R-CNN](https://arxiv.org/abs/1504.08083)
        :param x: An input Tensor to calculate smooth L1 loss.
        """
        square_loss = 0.5 * x ** 2
        absolute_loss = tf.abs(x)
        loss = tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss - 0.5)
        return loss

    def _ssd_loss(self, gloc, gcls):
        with tf.name_scope('hard_negtave_mining'):
            pos_mask_list, neg_mask_list = self.hard_negtave_mining(gcls)
        with tf.name_scope('losses'):
            with tf.name_scope('regularization'):
                tf.summary.scalar('regular', tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

            with tf.name_scope('cross_entropy'):
                xents = []
                for i in range(len(neg_mask_list)):
                    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.classification[i],
                                                                          labels=gcls[i])
                    xents.append(xent)

            with tf.name_scope('neg_cross_entropy'):
                loss_list = []
                for i, indices in enumerate(neg_mask_list):
                    loss = tf.cast(neg_mask_list[i], tf.float32) * xents[i]
                    loss = tf.reduce_mean(loss, axis=0)
                    loss = tf.reduce_sum(loss)
                    loss_list.append(loss)
                loss = tf.add_n(loss_list)
                tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
                tf.summary.scalar('neg_xent', loss)

            with tf.name_scope('pos_cross_entropy'):
                loss_list = []
                for i, indices in enumerate(neg_mask_list):
                    loss = tf.cast(pos_mask_list[i], tf.float32) * xents[i]
                    loss = tf.reduce_mean(loss, axis=0)
                    loss = tf.reduce_sum(loss)
                    loss_list.append(loss)
                loss = tf.add_n(loss_list)
                tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
                tf.summary.scalar('pos_xent', loss)


            with tf.name_scope('location_loss'):
                loc_loss = []
                for i in range(len(self.location)):
                    loss = self.smooth_l1_loss(gloc[i] - self.location[i])
                    loss = tf.reduce_sum(loss, -1)
                    loss = tf.cast(pos_mask_list[i], tf.float32) * loss
                    loss = tf.reduce_mean(loss, axis=0)
                    loss = tf.reduce_sum(loss)
                    loc_loss.append(loss)
                loss = tf.add_n(loc_loss)
                tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
                tf.summary.scalar('loc_loss', loss)

    def get_loss(self, gloc, gcls):
        self._ssd_loss(gloc, gcls)
        loss = tf.get_collection(tf.GraphKeys.LOSSES) + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n(loss)
        return loss

def main():
    x = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
    net = vgg16_ssd()
    net.build(x)
    net.get_loss()




if __name__ == '__main__':
    main()