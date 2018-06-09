import os

import cv2
import numpy as np
import tensorflow as tf

from inputs.ssdinputs import SSDInputs
from ssd.configure import Configure
from ssd.nets import ssdalexnet
from model_builder import ModelBuilder


# tf.enable_eager_execution()

config = Configure().get_config()

log_dir = '../log/ssd'
ckpt_dir = '../ckpt/ssd'
filenames = config['image']['path']

builder = ModelBuilder(config, mode='eval', input_class=SSDInputs)

net = builder.model

net_inputs = net.inputs
ground_truth = net.ground_truth
pipeline = builder.pipeline
# tf.summary.histogram('location', locations)
# xxx = tf.placeholder(dtype=tf.float32, shape=(config['train']['batch_size'], 300, 300, 3))
# net = ssdalexnet.SSDAlexNet(net_inputs, 3, ground_truth, anchor_config=anchor_config)

box, prob = pipeline.decode(net.outputs, 3)

mean = np.array([123, 117, 104])
mean = np.reshape(mean, [1, 1, 3])
# drawed = tf.image.draw_bounding_boxes((net_inputs['images']*128+mean)/255., tf.expand_dims(box, 0))


color = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
saver = tf.train.Saver(name="saver")
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    dir = tf.train.latest_checkpoint(ckpt_dir)
    saver.restore(sess, dir)


    while True:
        im, b, p = sess.run([net_inputs['images'], box, prob])
        im = (im * 128 + 128).astype(np.uint8)[0, :]
        for bb, pp, c in zip(b, p, color):
            for bbb, ppp in zip(bb, pp):
                cv2.rectangle(im, (int(bbb[1]*300), int(bbb[0]*300)), (int(bbb[3]*300), int(bbb[2]*300)), c, 2)
                print(ppp)
        cv2.imshow("", im)
        cv2.waitKey()
        a = 1
        # result = (sess.run(drawed)*255).astype(np.uint8)[0,:]
        # cv2.imshow("", result)
        # cv2.waitKey(0)

