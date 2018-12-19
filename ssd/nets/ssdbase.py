from abc import ABCMeta, abstractmethod
import six
import tensorflow as tf
import tensorflow.contrib.layers as layers

from basenets import Net
from basenets import utils

@six.add_metaclass(ABCMeta)
class SSDBase(Net):

    def __init__(self, *args, **kwargs):
        super(SSDBase, self).__init__(*args, **kwargs)
        self.outputs['classification'] = None
        self.outputs['location'] = None

    @abstractmethod
    def base_net(self):
        raise NotImplementedError

    @abstractmethod
    def down_sample(self):
        raise NotImplementedError

    @abstractmethod
    def extra_net(self):
        raise NotImplementedError

    def predict(self):
        feature_maps = [self.endpoints[f] for f in self.src]
        location_list = []
        classification_list = []
        with tf.contrib.framework.arg_scope([layers.conv2d],
                                            activation_fn=None,
                                            weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                            biases_regularizer=layers.l2_regularizer(self.weight_decay)):
            for i, feature_map in enumerate(feature_maps):
                # num_outputs = self.num_anchors[i] * self.A[i] * self.A[i] * (self.num_classes + 1 + 4)
                # prediction = layers.conv2d(feature_map, num_outputs, [3, 3], 1, scope='pred_%d' % i, activation_fn=None)
                #
                # prediction = utils.tf_re(prediction, self.A[i])
                num_outputs = self.num_anchors[i] * (self.num_classes + 1 + 4)
                feature_map = utils.tf_re(feature_map, self.A[i])
                prediction = layers.conv2d(feature_map, num_outputs, [3, 3], 1, scope='pred_%d' % i, activation_fn=None)

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
                location_list.append(locations)
                classification_list.append(classifications)
        self.outputs['location'] = location_list
        self.outputs['classification'] = classification_list

    def calc_loss(self):
        location = self.outputs['location']
        classification = self.outputs['classification']
        batch_sise = classification[0].get_shape().as_list()[0]

        location = tf.unstack(tf.concat([tf.reshape(t, [batch_sise, -1,  4]) for t in location], axis=1))
        l_gt = self.ground_truth['locations']
        l_gt = tf.unstack(tf.concat([tf.reshape(t, [batch_sise, -1, 4]) for t in l_gt], axis=1))

        cls_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                for (logits, labels) in zip(classification, self.ground_truth['labels'])]
        cls_loss = tf.unstack(tf.concat([layers.flatten(t) for t in cls_loss], axis=-1))

        flattened_label = tf.concat([layers.flatten(t) for t in self.ground_truth['labels']], axis=-1)

        flattened_label_list = tf.unstack(flattened_label)

        num_pos_l = []

        cls_loss_pos = []
        cls_loss_neg = []
        loc_loss_pos = []
        for i, cls in enumerate(cls_loss):
            sorted_cls_loss = tf.contrib.framework.sort(cls, -1, 'DESCENDING')
            labels = flattened_label_list[i]
            pos = tf.greater(labels, 0)
            neg = tf.equal(labels, 0)
            num_pos = tf.reduce_sum(tf.to_int32(pos))
            num_neg = 3 * num_pos
            max_neg = tf.reduce_sum(tf.to_int32(neg))
            max_idx = tf.minimum(num_neg - 1, max_neg - 1)
            min_score = sorted_cls_loss[max_idx]
            selected = tf.greater(cls, min_score)
            neg = tf.logical_and(neg, selected)

            ass = tf.assert_none_equal(num_pos, 0)
            tf.add_to_collection(ass, "ASSERT")

            loc = tf.boolean_mask(location[i], pos)
            target = tf.boolean_mask(l_gt[i], pos)

            loc = tf.losses.huber_loss(target, loc, reduction='none')
            cls_loss_pos.append(tf.reduce_sum(tf.boolean_mask(cls, pos)))
            cls_loss_neg.append(tf.reduce_sum(tf.boolean_mask(cls, neg)))
            loc_loss_pos.append(tf.reduce_sum(loc))
            num_pos_l.append(num_pos)

        num_pos = tf.to_float(tf.add_n(num_pos_l))

        loc_loss = tf.add_n(loc_loss_pos) / num_pos
        cls_loss_pos = tf.add_n(cls_loss_pos) / num_pos
        cls_loss_neg = tf.add_n(cls_loss_neg) / num_pos
        num_pos /= batch_sise

        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.add_n(reg)

        total_loss = loc_loss + cls_loss_pos + cls_loss_neg + reg_loss

        with tf.variable_scope('loss'):
            tf.summary.scalar('loc_loss', loc_loss)
            tf.summary.scalar('cls_loss_pos', cls_loss_pos)
            tf.summary.scalar('cls_loss_neg', cls_loss_neg)
            tf.summary.scalar('reg_loss', reg_loss)
            tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('num_pos', num_pos)
        return total_loss