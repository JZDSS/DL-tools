from __future__ import division
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from nets import net
import numpy as np

NEGTAVE_EROSS_ENTROPY = 'neg_xent'
POSITIVE_EROSS_ENTROPY = 'pos_xent'
LOCATION_LOSS = 'loc_loss'

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


class SSDNet(net.Net):

    def __init__(self, anchor_scales=default_anchor_scales, aspect_ratios=default_aspect_ratios,
                 ext_anchors=default_ext_anchor_scales, *args, **kwargs):
        super(SSDNet, self).__init__(*args, **kwargs)
        self.anchor_scales = anchor_scales
        self.aspect_ratios = aspect_ratios
        self.num_anchors = [len(ratio) + 2 for ratio in self.aspect_ratios]
        self.ext_anchors = ext_anchors
        self.feature_map_size = []
        self.feature_maps = None

    def l2_norm(self, x, n):
        x = tf.nn.l2_normalize(x, [0, 1, 2, 3])
        gamma = 20 * tf.get_variable('gamma', shape=[n], initializer=tf.ones_initializer(tf.float32), trainable=True)
        x = x * gamma
        return x

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
            val, _ = tf.nn.top_k(-flat_neg, num_negtave)
            maxval = -val[-1]
            neg_mask = tf.less(flat_neg, maxval)
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
                tf.add_to_collection(NEGTAVE_EROSS_ENTROPY, loss)
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
                tf.add_to_collection(POSITIVE_EROSS_ENTROPY, loss)
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
                tf.add_to_collection(LOCATION_LOSS, loss)
                tf.summary.scalar('loc_loss', loss)

    def get_loss(self, gloc, gcls, scope):
        self._ssd_loss(gloc, gcls)
        loss = tf.get_collection(tf.GraphKeys.LOSSES, scope=scope) + \
               tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
        loss = tf.add_n(loss)
        return loss
