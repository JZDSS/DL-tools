import os

import cv2
import numpy as np
import tensorflow as tf

from inputs.ssdinputs import SSDInputs
from ssd.configure import Configure
from ssd.nets import ssdalexnet

config = Configure().get_config()
anchor_config = config['anchor']
feature_map_size = anchor_config['feature_map_size']
anchor_scales = anchor_config['scales']
ext_anchor_scales = anchor_config['extra_scales']
aspect_ratios = anchor_config['aspect_ratios']

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
filenames = config['image']['path']
pipeline = SSDInputs(config, fake=False)
net_inputs, ground_truth = pipeline.input_pipeline(filenames, len(filenames), 1)
# tf.summary.histogram('location', locations)
# xxx = tf.placeholder(dtype=tf.float32, shape=(config['train']['batch_size'], 300, 300, 3))
net = ssdalexnet.SSDAlexNet(net_inputs, 3, ground_truth, anchor_config=anchor_config)

box, prob = pipeline.decode(net.outputs)
selected_indices = tf.image.non_max_suppression(box, prob, 10, 0.4)
box = tf.gather(box, selected_indices)
prob = tf.gather(prob, selected_indices)
mean = np.array([123, 117, 104])
mean = np.reshape(mean, [1, 1, 3])
drawed = tf.image.draw_bounding_boxes((net_inputs['images']*128+mean)/255., tf.expand_dims(box, 0))

saver = tf.train.Saver(name="saver")
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    dir = tf.train.latest_checkpoint(ckpt_dir)
    saver.restore(sess, dir)



    # image = image / 128.0
    while True:
        # im = sess.run(net_inputs['images'])
        result = (sess.run(drawed)*255).astype(np.uint8)[0,:]
        cv2.imshow("", result)
        cv2.waitKey(0)

