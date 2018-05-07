from ssd.anchor import *
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ssd import ssd_input
from ssd.nets import ssd_alexnet
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
log_dir = '../log/ssd'
ckpt_dir = '../ckpt/ssd'
images, locations, labels = ssd_input.input_pipeline(
    tf.train.match_filenames_once(os.path.join('../ssd', '*.tfrecords')), 10, read_threads=1)
# tf.summary.histogram('location', locations)
xxx = tf.placeholder(dtype=tf.float32, shape=(None, 300, 300, 3))
net = ssd_alexnet.SSD_AlexNet(xxx, 3, {'locations':locations, 'labels':labels})
saver = tf.train.Saver(name="saver")
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    dir = tf.train.latest_checkpoint(ckpt_dir)
    saver.restore(sess, dir)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    mean = np.array([123, 117, 104])
    mean = np.reshape(mean, [1, 1, 3])

    # image = image / 128.0
    while True:
        im = sess.run(images)
        locs = sess.run(net.location, feed_dict={xxx: im})
        prob = sess.run([tf.nn.softmax(c, axis=-1) for c in net.classification], feed_dict={xxx: im})
        labs = sess.run([tf.argmax(c, axis=-1) for c in net.classification], feed_dict={xxx: im})
        # print(locs[0])

        im = (im[0, :, :, :]*128+mean).astype(np.uint8)
        # plt.imshow(im)
        # plt.show()
        for n_map, lab in enumerate(labs):
            lab = lab[0, :, :, :]
            p = prob[n_map][0, :, :, :]
            for c in range(lab.shape[-1]):
                if c == 0:
                    continue
                labb = lab[:,:,c]
                pp = p[:,:,c]
                for y in range(labb.shape[0]):
                    for x in range(labb.shape[1]):
                        if labb[y, x] != 0 and pp[y, x, labb[y, x]] > 0.7:
                            print(labb[y, x], pp[y, x, labb[y, x]])
                            bbox = locs[n_map][0, y, x, c,:]  #[cx, cy, w, h]
                            # print(bbox)
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

        # plt.imshow(im)
        # plt.show()
        cv2.imshow('', im)
        cv2.waitKey(0)
    coord.request_stop()
    coord.join(threads)