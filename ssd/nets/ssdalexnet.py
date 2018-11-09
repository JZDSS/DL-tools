import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from basenets import alexnet
from basenets import utils
from ssd.nets import ssdbase

class SSDAlexNet(alexnet.AlexNet, ssdbase.SSDBase):

    def __init__(self,
                 inputs,
                 num_classes,
                 ground_truth,
                 anchor_config,
                 name='SSD_AlexNet',
                 npy_path=None,
                 weight_decay=0.0001,
                 **kwargs):
        super(SSDAlexNet, self).__init__(inputs, name, npy_path, weight_decay=weight_decay,
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
        del self.endpoints['fc6']
        del self.endpoints['fc7']
        del self.endpoints['fc8']
        self.down_sample()

    def down_sample(self):
        endpoints = self.endpoints
        y = endpoints['conv5']
        with tf.contrib.framework.arg_scope([layers.conv2d],
                       weights_regularizer=layers.l2_regularizer(self.weight_decay),
                       biases_regularizer=layers.l2_regularizer(self.weight_decay)):
            with tf.variable_scope('rebuild'):
                y = layers.conv2d(y, 1024, [3, 3], 1, 'SAME', rate=6, scope='fc6')
                endpoints['fc6'] = y
                y = layers.conv2d(y, 1024, [1, 1], 1, 'VALID', scope='fc7')
                endpoints['fc7'] = y

    def extra_net(self):
        endpoints = self.endpoints
        with tf.contrib.framework.arg_scope([layers.conv2d],
                                            weights_regularizer=layers.l2_regularizer(self.weight_decay),
                                            biases_regularizer=layers.l2_regularizer(self.weight_decay)):
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
        feature_maps = [self.endpoints[f] for f in self.src]
        location_list = []
        classification_list = []
        with tf.contrib.framework.arg_scope([layers.conv2d],
                                            activation_fn=None,
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
        batch_sise = classification[0].get_shape().as_list()[0]

        location = tf.unstack(tf.concat([tf.reshape(t, [batch_sise, -1,  4]) for t in location], axis=1))
        l_gt = self.ground_truth['locations']
        l_gt = tf.unstack(tf.concat([tf.reshape(t, [batch_sise, -1, 4]) for t in l_gt], axis=1))

        cls_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                for (logits, labels) in zip(classification, self.ground_truth['labels'])]
        cls_loss = tf.unstack(tf.concat([layers.flatten(t) for t in cls_loss], axis=-1))

        # loc_loss = [tf.reduce_sum(tf.losses.huber_loss(target, loc, reduction='none'), axis=-1)
        #             for (target, loc) in zip(self.ground_truth['locations'], location)]
        # loc_loss = tf.unstack(tf.concat([layers.flatten(t) for t in loc_loss], axis=-1))

        flattened_label = tf.concat([layers.flatten(t) for t in self.ground_truth['labels']], axis=-1)

        # cls_loss_list = tf.unstack(cls_loss)
        # loc_loss_list = tf.unstack(loc_loss)
        flattened_label_list = tf.unstack(flattened_label)

        poss = []
        negs = []
        num_pos_l = []
        # num_neg_l = []
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
            # cls = cls
            a = location[i]
            loc = tf.boolean_mask(location[i], pos)
            target = tf.boolean_mask(l_gt[i], pos)

            loc = tf.losses.huber_loss(target, loc, reduction='none')
            # pos = tf.greater(pos, 0)
            # neg = tf.greater(neg, 0)
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





if __name__ == '__main__':
    from ssd import ssd_input
    import os
    import time

    log_dir = '../../log/ssd'
    ckpt_dir = '../../ckpt/ssd'
    images, locations, labels = ssd_input.input_pipeline(
        tf.train.match_filenames_once(os.path.join('../../ssd', '*.tfrecords')), 8, read_threads=1)
    tf.summary.image('image', images, 1)
    # tf.summary.histogram('location', locations)
    net = SSDAlexNet(images, 3, {'locations':locations, 'labels':labels}, npy_path='../../npy/alexnet.npy')
    loss = net.get_loss()

    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.minimize(loss)
    saver = tf.train.Saver(name="saver", max_to_keep=10)
    for i, l in enumerate(locations):
        pl = net.location[i]
        tf.summary.histogram('gtl%d'%i, l)
        tf.summary.histogram('prl%d'%i, pl)
    summ = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        i = 0
        if os.listdir(ckpt_dir):
            print('train from ckpt')
            dir = tf.train.latest_checkpoint(ckpt_dir)
            i = int(dir.split('-')[-1])
            saver.restore(sess, dir)
        else:
            print('train from alexnet')
            sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        train_writer.flush()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        ll = sess.run(net.location)
        d = 100

        while True:
            if i % d == 0:
                saver.save(sess, os.path.join(ckpt_dir, 'ssd_model'), global_step=i)
                summary = sess.run(summ)
                train_writer.add_summary(summary, i)
                train_writer.flush()
                if i != 0:
                    time.sleep(10)
            sess.run(train_op)
            i += 1

        coord.request_stop()
        coord.join(threads)

# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from ssd import ssd_input
# anchors = []
# for i, size in enumerate(feature_map_size):
#     # d for default boxes(anchors)
#     # x and y coordinate of centers of every cell, normalized to [0, 1]
#     d_cy, d_cx = np.mgrid[0:size[0], 0:size[1]].astype(np.float32)
#     d_cx = (d_cx + 0.5) / size[1]
#     d_cy = (d_cy + 0.5) / size[0]
#     d_cx = np.expand_dims(d_cx, axis=-1)
#     d_cy = np.expand_dims(d_cy, axis=-1)
#
#     # calculate width and heights
#     d_w = []
#     d_h = []
#     scale = anchor_scales[i]
#     # two aspect ratio 1 anchor scales
#     d_w.append(ext_anchor_scales[i])
#     d_w.append(scale)
#     d_h.append(ext_anchor_scales[i])
#     d_h.append(scale)
#     # other anchor scales
#     for ratio in aspect_ratios[i]:
#         d_w.append(scale * np.sqrt(ratio))
#         d_h.append(scale / np.sqrt(ratio))
#     d_w = np.array(d_w, dtype=np.float32)
#     d_h = np.array(d_h, dtype=np.float32)
#
#     d_ymin = d_cy - d_h / 2
#     d_ymax = d_cy + d_h / 2
#     d_xmin = d_cx - d_w / 2
#     d_xmax = d_cx + d_w / 2
#
#     d_h = d_ymax - d_ymin
#     d_w = d_xmax - d_xmin
#     d_cx = (d_xmax + d_xmin) / 2
#     d_cy = (d_ymax + d_ymin) / 2
#     anchors.append(np.stack([d_cx, d_cy, d_w, d_h], -1))
# log_dir = '../../log/ssd'
# ckpt_dir = '../../ckpt/ssd'
# images, locations, labels = ssd_input.input_pipeline(
#     tf.train.match_filenames_once(os.path.join('../../ssd', '*.tfrecords')), 1, read_threads=1)
# # tf.summary.histogram('location', locations)
# xxx = tf.placeholder(dtype=tf.float32, shape=(None, 300, 300, 3))
# net = SSD_AlexNet(xxx, 20, {'locations':locations, 'labels':labels}, npy_path='../../npy/alexnet.npy')
# saver = tf.train.Saver(name="saver")
# with tf.Session() as sess:
#     sess.run(tf.local_variables_initializer())
#     dir = tf.train.latest_checkpoint(ckpt_dir)
#     saver.restore(sess, dir)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     while True:
#         im = sess.run(images)
#         locs = sess.run(net.location, feed_dict={xxx: im})
#         labs = sess.run([tf.argmax(c, axis=-1) for c in net.classification], feed_dict={xxx: im})
#         print(locs[0])
#         im = im[0, :, :, :].astype(np.uint8)
#         # plt.imshow(im)
#         # plt.show()
#         for n_map, lab in enumerate(labs):
#             lab = lab[0, :, :, :]
#             for c in range(lab.shape[-1]):
#                 labb = lab[:,:,c]
#                 for y in range(labb.shape[0]):
#                     for x in range(labb.shape[1]):
#                         if labb[y, x] != 0:
#
#                             bbox = locs[n_map][0, y, x, c,:]  #[cx, cy, w, h]
#                             print(bbox)
#                             d_cx = anchors[n_map][y, x, c, 0]
#                             d_cy = anchors[n_map][y, x, c, 1]
#                             d_w = anchors[n_map][y, x, c, 2]
#                             d_h = anchors[n_map][y, x, c, 3]
#                             bbox[0] = bbox[0] * d_w + d_cx
#                             bbox[1] = bbox[1] * d_h + d_cy
#                             bbox[2] = np.exp(bbox[2]) * d_w
#                             bbox[3] = np.exp(bbox[3]) * d_h
#                             minx = int((bbox[0] - bbox[2]/2)*300)
#                             maxx = int((bbox[0] + bbox[2]/2)*300)
#                             miny = int((bbox[1] - bbox[3]/2)*300)
#                             maxy = int((bbox[1] + bbox[3]/2)*300)
#                             cv2.rectangle(im, (minx, miny), (maxx, maxy), (0,0,255), 1)
#
#         # plt.imshow(im)
#         # plt.show()
#         cv2.imshow('', im)
#         cv2.waitKey(0)
#     coord.request_stop()
#     coord.join(threads)