import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from inputs.ssdinputs import SSDInputs
from ssd.configure import Configure
from builder.model_builder import ModelBuilder
import ssd.dataset.voc_eval as voc

config = Configure(path='../ssd/ssd_mobile.config').get_config()

log_dir = config['eval']['log_dir']
ckpt_dir = config['eval']['ckpt_dir']
# filenames = config['image']['path']

builder = ModelBuilder(config, mode='eval', input_class=SSDInputs)

net = builder.model

net_inputs = net.inputs
ground_truth = net.ground_truth
box, prob = builder.pipeline.decode_for_ap(net.outputs, config['model']['num_classes'])

color = [(255, 0, 0)]
saver = tf.train.Saver(name="saver")
num_classes = config['model']['num_classes'] + 1

# txt = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#     'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#     'dog', 'horse', 'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor']
txt = ['fish']
fs = []
for i in range(len(txt)):
    fs.append(open('{}_det.txt'.format(txt[i]), 'w'))
s = 1
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    dir = tf.train.latest_checkpoint(ckpt_dir)
    saver.restore(sess, dir)

    while True:
        try:
            im, name, b, p = sess.run([net_inputs['images'], net_inputs['names'], box, prob], feed_dict={net.is_training: False})
            im = (im * 128 + 128).astype(np.uint8)[0, :]
            im = cv2.resize(im, (500, 375))
            name = name[0].decode()
            for bb, pp, f in zip(b, p, fs):
                for bbb, ppp in zip(bb, pp):
                    xmin = max(int(bbb[1] * 500) + 1, 0)
                    ymin = max(int(bbb[0] * 375) + 1, 0)
                    xmax = min(int(bbb[3] * 500) + 1, 499)
                    ymax = min(int(bbb[2] * 375) + 1, 499)
                    xc = (xmin + xmax) / 2
                    yc = (ymin + ymax) / 2
                    w = (xmax - xmin) / 2 * s
                    h = (ymax - ymin) / 2 * s
                    xmin = int(xc - w)
                    xmax = int(xc + w)
                    ymin = int(yc - h)
                    ymax = int(yc + h)
                    if ppp > 0.11:
                        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color[0], 2)
                    f.write('{:s} {:f} {:d} {:d} {:d} {:d}\n'.format(name, ppp, xmin, ymin, xmax, ymax))
                    a = 1
            cv2.imshow("", im)
            cv2.waitKey(1)
        except tf.errors.OutOfRangeError as e:
            for f in fs:
                f.close()
            break
mAP = 0
for n in txt:
    rec, prec, ap = voc.voc_eval("/home/yqi/workspace/DL-tools/test/{}_det.txt".format(n),
                                 "/home/yqi/Desktop/WineDownloads/大作业3/to student/FishData_subset/Annotations/{:s}.xml",
                                 "/home/yqi/Desktop/WineDownloads/大作业3/to student/FishData_subset/ImageSets/Main/test.txt",
                                 n,
                                 ".",
                                 ovthresh=0.5,
                                 use_07_metric=False)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlim(0, 1)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.ylim(0, 1)
    plt.grid()
    plt.text(0.1, 1.02, n, fontsize=14)
    plt.text(0.82, 1.02, 'AP=%.02f%%' % (ap * 100), fontsize=14)
    plt.savefig(n+'.png')

    mAP += ap / len(txt)
    print('AP for {}:'.format(n), ap)
print("mAP:", mAP)
