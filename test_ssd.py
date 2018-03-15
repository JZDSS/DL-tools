import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
from nets import ssd_input, vgg16_ssd
import tensorflow as tf


with tf.device('/cpu:0'):
    # s_k = s_min + (s_max - s_min)/(m - 1)(k - 1)
    anchor_scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]          # of original image size
    # s_k^{'} = sqrt(s_k*s_k+1)
    ext_anchor_scales = [0.26, 0.28, 0.43, 0.60, 0.78, 0.98]    # of original image size
    # omitted aspect ratio 1
    aspect_ratios = [[1 / 2, 2],            # conv4_3
                     [1 / 3, 1 / 2, 2, 3],  # conv7
                     [1 / 3, 1 / 2, 2, 3],  # conv8_2
                     [1 / 3, 1 / 2, 2, 3],  # conv9_2
                     [1 / 2, 2],            # conv10_2
                     [1 / 2, 2]]            # conv11_2
    feature_map_size = [[38, 38],
                        [19, 19],
                        [10, 10],
                        [5, 5],
                        [3, 3],
                        [1, 1]]

    anchors = []
    for i, size in enumerate(feature_map_size):
        # d for default boxes(anchors)
        # x and y coordinate of centers of every cell, normalized to [0, 1]
        d_cy, d_cx = np.mgrid[0:size[0], 0:size[1]].astype(np.float32)
        d_cx = (d_cx + 0.5) / size[1]
        d_cy = (d_cy + 0.5) / size[0]
        d_cx = np.expand_dims(d_cx, axis=-1)
        d_cy = np.expand_dims(d_cy, axis=-1)

        # calculate width and heights
        d_w = []
        d_h = []
        scale = anchor_scales[i]
        # two aspect ratio 1 anchor scales
        d_w.append(ext_anchor_scales[i])
        d_w.append(scale)
        d_h.append(ext_anchor_scales[i])
        d_h.append(scale)
        # other anchor scales
        for ratio in aspect_ratios[i]:
            d_w.append(scale * np.sqrt(ratio))
            d_h.append(scale / np.sqrt(ratio))
        d_w = np.array(d_w, dtype=np.float32)
        d_h = np.array(d_h, dtype=np.float32)

        d_ymin = d_cy - d_h / 2
        d_ymax = d_cy + d_h / 2
        d_xmin = d_cx - d_w / 2
        d_xmax = d_cx + d_w / 2

        d_h = d_ymax - d_ymin
        d_w = d_xmax - d_xmin
        d_cx = (d_xmax + d_xmin) / 2
        d_cy = (d_ymax + d_ymin) / 2
        anchors.append(np.stack([d_cx, d_cy, d_w, d_h], -1))

    i_and_l = ssd_input.input_pipeline(
        tf.train.match_filenames_once(os.path.join('nets/ssd', '*.tfrecords')), 1, read_threads=1)
    images = i_and_l[0]
    locations = i_and_l[1:len(i_and_l) // 2 + 1]
    labels = i_and_l[len(i_and_l)//2 + 1:]

    net = vgg16_ssd.vgg16_ssd()
    net.build(images)

    locations = net.location
    labels = net.classification

    prob = [tf.nn.softmax(label, -1) for label in labels]

    labels = [tf.argmax(label, -1) for label in labels]
    saver = tf.train.Saver(name="saver")

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./ckpt0'))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        while True:
            im, locs, labs, p = sess.run([images, locations, labels, prob])
            # print(locs[0])
            im = im[0, :, :, :].astype(np.uint8)
            plt.imshow(im)
            cv2.imshow('', im)
            cv2.waitKey(0)
            plt.show()
            for n_map, lab in enumerate(labs):
                lab = lab[0, :, :, :]
                for c in range(lab.shape[-1]):
                    labb = lab[:,:,c]
                    for y in range(labb.shape[0]):
                        for x in range(labb.shape[1]):
                            if labb[y, x] != 0 and p[n_map][0, y, x, c, labb[y, x]] > 0.999:

                                bbox = locs[n_map][0, y, x, c,:]  #[cx, cy, w, h]
                                print(bbox)
                                d_cx = anchors[n_map][y, x, c, 0]
                                d_cy = anchors[n_map][y, x, c, 1]
                                d_w = anchors[n_map][y, x, c, 2]
                                d_h = anchors[n_map][y, x, c, 3]
                                bbox[0] = bbox[0] * d_w + d_cx
                                bbox[1] = bbox[1] * d_h + d_cy
                                bbox[2] = np.exp(bbox[2]) * d_w
                                bbox[3] = np.exp(bbox[3]) * d_h
                                minx = int((bbox[0] - bbox[2]/2)*300)
                                maxx = int((bbox[0] + bbox[2]/2)*300)
                                miny = int((bbox[1] - bbox[3]/2)*300)
                                maxy = int((bbox[1] + bbox[3]/2)*300)
                                cv2.rectangle(im, (minx, miny), (maxx, maxy), (0,0,255), 1)

                                plt.imshow(im)
                                plt.show()
        coord.request_stop()
        coord.join(threads)