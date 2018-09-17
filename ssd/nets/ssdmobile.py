import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope
import numpy as np

from basenets import mobilenet
from basenets import utils
from ssd.nets import ssdbase


class SSDMobileNet(mobilenet.MobileNet, ssdbase.SSDBase):

    def __init__(self,
                 inputs,
                 num_classes,
                 ground_truth,
                 anchor_config,
                 name='SSD_MobileNet',
                 npy_path=None,
                 weight_decay=0.00004,
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
            endpoints['conv2'] = y
            y = layers.conv2d(y, 128, (1, 1), 1, scope='conv3')
            endpoints['conv3'] = y
            y = layers.conv2d(y, 256, (3, 3), 2, scope='conv4')
            endpoints['conv4'] = y
            y = layers.conv2d(y, 128, (1, 1), 1, scope='conv5')
            endpoints['conv5'] = y
            y = layers.conv2d(y, 256, (3, 3), 2, scope='conv6')
            endpoints['conv6'] = y
            y = layers.conv2d(y, 64, (1, 1), 1, scope='conv7')
            endpoints['conv7'] = y
            y = layers.conv2d(y, 128, (3, 3), 2, scope='conv8')
            endpoints['conv8'] = y

    def predict(self):
        feature_maps = [self.endpoints[f] for f in self.src]
        location_list = []
        classification_list = []
        with tf.contrib.framework.arg_scope([layers.conv2d],
                                            weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                            biases_regularizer=layers.l2_regularizer(self.weight_decay)):
            for i, feature_map in enumerate(feature_maps):
                num_outputs = self.num_anchors[i] * (self.num_classes + 1 + 4)
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

    def calc_loss(self):
        location = self.outputs['location']
        classification = self.outputs['classification']
        cls_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                for (logits, labels) in zip(classification, self.ground_truth['labels'])]
        cls_loss = tf.concat([layers.flatten(t) for t in cls_loss], axis=-1)

        loc_loss = [tf.reduce_sum(tf.losses.huber_loss(target, loc, reduction='none'), axis=-1)
                    for (target, loc) in zip(self.ground_truth['locations'], location)]
        loc_loss = tf.concat([layers.flatten(t) for t in loc_loss], axis=-1)

        flattened_label = tf.concat([layers.flatten(t) for t in self.ground_truth['labels']], axis=-1)

        cls_loss_list = tf.unstack(cls_loss)
        # loc_loss_list = tf.unstack(loc_loss)
        flattened_label_list = tf.unstack(flattened_label)

        masks = []
        poss = []
        negs = []
        for i, cls in enumerate(cls_loss_list):
            sorted_cls_loss = tf.contrib.framework.sort(cls, -1, 'DESCENDING')
            labels = flattened_label_list[i]
            pos = tf.greater(labels, 0)
            neg = tf.equal(labels, 0)
            num_pos = tf.reduce_sum(tf.to_int32(pos))
            max_neg = tf.reduce_sum(tf.to_int32(neg))
            max_idx = tf.minimum(3 * num_pos, max_neg - 1)
            num_selected = num_pos + max_idx + 1
            min_score = sorted_cls_loss[max_idx]
            selected = tf.greater(cls, min_score)
            neg = tf.logical_and(neg, selected)

            mask = tf.logical_or(pos, neg)
            ass = tf.assert_none_equal(num_pos, 0)
            tf.add_to_collection(ass, "ASSERT")
            # masks.append(tf.to_float(mask))
            # poss.append(tf.to_float(pos))
            # negs.append(tf.to_float(neg))
            masks.append(utils.safe_division(tf.to_float(mask), tf.to_float(num_pos)))
            poss.append(utils.safe_division(tf.to_float(pos), tf.to_float(num_pos)))
            negs.append(utils.safe_division(tf.to_float(neg), tf.to_float(num_pos)))

        cls_mask = tf.stack([m for m in masks], axis=0)
        pos_mask = tf.stack([m for m in poss], axis=0)
        neg_mask = tf.stack([m for m in negs], axis=0)

        cls_loss_pos = tf.multiply(cls_loss, pos_mask)
        cls_loss_pos = tf.reduce_sum(cls_loss_pos, axis=-1)
        cls_loss_pos = tf.reduce_mean(cls_loss_pos)

        cls_loss_neg = tf.multiply(cls_loss, neg_mask)
        cls_loss_neg = tf.reduce_sum(cls_loss_neg, axis=-1)
        cls_loss_neg = tf.reduce_mean(cls_loss_neg)

        loc_loss = tf.multiply(loc_loss, pos_mask)
        loc_loss = tf.reduce_sum(loc_loss, axis=-1)
        loc_loss = tf.reduce_mean(loc_loss)

        tf.summary.scalar('loc_loss', loc_loss)
        tf.summary.scalar('cls_loss_pos', cls_loss_pos)
        tf.summary.scalar('cls_loss_neg', cls_loss_neg)

        total_loss = loc_loss + cls_loss_pos + cls_loss_neg
        tf.summary.scalar('total_loss', total_loss)
        return total_loss

