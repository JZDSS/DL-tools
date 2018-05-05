import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from nets import alexnet
from nets import utils
from ssd.anchor import *

class SSD_AlexNet(alexnet.AlexNet):

    def __init__(self,
                 image,
                 num_classes,
                 ground_truth,
                 name='SSD_AlexNet',
                 npy_path=None,
                 weight_decay=0.0004,
                 feature_from=['conv2', 'fc7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']):
        super(SSD_AlexNet, self).__init__(image, name, npy_path)
        self.ground_truth = ground_truth
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.feature_from = feature_from
        self.aspect_ratios = aspect_ratios
        self.num_anchors = [len(ratio) + 2 for ratio in self.aspect_ratios]
        self.ext_anchors = ext_anchor_scales
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
        y = endpoints['conv5']
        with tf.variable_scope('rebuild'):
            y = utils.atrous_conv2d(y, 1024, [3, 3], 6, 'SAME', scope='fc6')
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
        y = layers.conv2d(y, 256, [1, 1], 1, 'SAME', scope='conv8_1')
        endpoints['conv8_1'] = y
        y = layers.conv2d(y, 512, [3, 3], 2, 'SAME', scope='conv8_2')
        endpoints['conv8_2'] = y
        y = layers.conv2d(y, 128, [1, 1], 1, 'SAME', scope='conv9_1')
        endpoints['conv9_1'] = y
        y = layers.conv2d(y, 256, [3, 3], 2, 'SAME', scope='conv9_2')
        endpoints['conv9_2'] = y
        y = layers.conv2d(y, 128, [1, 1], 1, 'SAME', scope='conv10_1')
        endpoints['conv10_1'] = y
        y = layers.conv2d(y, 256, [3, 3], 1, 'VALID', scope='conv10_2')
        endpoints['conv10_2'] = y
        y = layers.conv2d(y, 128, [1, 1], 1, 'SAME', scope='conv11_1')
        endpoints['conv11_1'] = y
        y = layers.conv2d(y, 256, [3, 3], 1, 'VALID', scope='conv11_2')
        endpoints['conv11_2'] = y

    def predict(self):
        feature_maps = [self.endpoints[f] for f in self.feature_from]
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

    def get_loss(self):
        cls_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                for (logits, labels) in zip(self.classification, self.ground_truth['labels'])]
        cls_loss = tf.concat([layers.flatten(t) for t in cls_loss], axis=-1)

        loc_loss = [tf.reduce_sum(tf.losses.huber_loss(target, loc, reduction='none'), axis=-1)
                    for (target, loc) in zip(self.ground_truth['locations'], self.location)]
        loc_loss = tf.concat([layers.flatten(t) for t in loc_loss], axis=-1)

        flattened_label = tf.concat([layers.flatten(t) for t in self.ground_truth['labels']], axis=-1)

        cls_loss_list = tf.unstack(cls_loss)
        # loc_loss_list = tf.unstack(loc_loss)
        flattened_label_list = tf.unstack(flattened_label)

        masks = []
        poss = []
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
            masks.append(utils.safe_division(tf.to_float(mask), tf.to_float(num_selected)))
            poss.append(utils.safe_division(tf.to_float(pos), tf.to_float(num_pos)))

        cls_mask = tf.stack([m for m in masks], axis=0)
        loc_mask = tf.stack([m for m in poss], axis=0)

        cls_loss = tf.multiply(cls_loss, cls_mask)
        cls_loss = tf.reduce_sum(cls_loss, axis=-1)
        cls_loss = tf.reduce_mean(cls_loss)

        loc_loss = tf.multiply(loc_loss, loc_mask)
        loc_loss = tf.reduce_sum(loc_loss, axis=-1)
        loc_loss = tf.reduce_mean(loc_loss)

        total_loss = loc_loss + cls_loss

        return total_loss

if __name__ == '__main__':
    from ssd import ssd_input
    import os
    import time
    images, locations, labels = ssd_input.input_pipeline(
        tf.train.match_filenames_once(os.path.join('../../ssd', '*.tfrecords')), 2, read_threads=1)
    net = SSD_AlexNet(images, 20, {'locations':locations, 'labels':labels}, npy_path='../../npy/alexnet.npy')
    loss = net.get_loss()
    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        d = 1
        i = 0
        while True:
            if i % d == 0:
                _, l = sess.run([train_op, loss])
                print(i, l)
            else:
                sess.run(train_op)
            i += 1

        coord.request_stop()
        coord.join(threads)