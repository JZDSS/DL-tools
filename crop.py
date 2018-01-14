# coding:utf-8
from __future__ import print_function
import tensorflow as tf
import os
from utils.patch import get_patches
import matplotlib.pyplot as plt
import numpy as np


flags = tf.app.flags

flags.DEFINE_string('data_dir', '/data/SPCup_preprocess', 'Data direction')
flags.DEFINE_string('out_dir', '/data/qiyao/valid_imgs', 'Output direction')
flags.DEFINE_string('meta_dir', '/data/qiyao/official/meta', '')
flags.DEFINE_string('out_file', './crop-out.txt', '')
flags.DEFINE_string('extra', '', 'j70, j90, g0_8 etc')

FLAGS = flags.FLAGS


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):

    dirs = os.listdir(FLAGS.data_dir)
    ff = open(FLAGS.out_file, 'w')
    if not ff:
        raise RuntimeError('OUTPUT FILE OPEN ERROR!!!!!!')
    tf.gfile.MakeDirs(FLAGS.out_dir)
    f = open(os.path.join(FLAGS.meta_dir, 'spc_classes.txt'), 'r')
    meta = {}
    line = f.readline()
    while line:
        label, class_name = line.split(' ')
        class_name = class_name[0:-1]
        meta[int(label)] = class_name
        line = f.readline()
    f.close()
    truth = []
    for data_dir in dirs:

        format = 'jpg' if 'jpeg' in data_dir or 'origin' in data_dir else 'tif'
        if 'jpeg70' in data_dir:
            extra = 'j70'
        elif 'jpeg90' in data_dir:
            extra = 'j90'
        elif 'gamma0.8' in data_dir:
            extra = 'g0_8'
        elif 'gamma1.2' in data_dir:
            extra = 'g1_2'
        elif 'resize0.5' in data_dir:
            extra = 'r0_5'
        elif 'resize0.8' in data_dir:
            extra = 'r0_8'
        elif 'resize1.5' in data_dir:
            extra = 'r1_5'
        elif 'resize2' in data_dir:
            extra = 'r2_0'
        elif 'raw' in data_dir:
            continue

        data_dir = os.path.join(FLAGS.data_dir, data_dir)

        def save_crop(sett):
            f = open(os.path.join(FLAGS.meta_dir, sett) + '.txt', 'r')
            image_names = []
            labels = []
            line = f.readline()
            while line:
                image_name, label = line.split(' ')
                label = label[0:-1]
                image_names.append(image_name + '-' + extra + '.' + format)
                labels.append(int(label))
                line = f.readline()
            f.close()
            for i, img_name in enumerate(image_names):
                full_path = os.path.join(data_dir, meta[labels[i]], img_name)
                print('processing ' + full_path, file=ff)
                ff.flush()
                img = plt.imread(full_path)
                for patch in get_patches(img, 1, 512):
                    plt.imsave(os.path.join(FLAGS.out_dir, meta[labels[i]] + '_' + img_name.split('.')[0]) + '.png', patch)
                    truth.append(labels[i])
        save_crop('valid')
        np.save('truth.npy', truth)


    ff.close()


if __name__ == "__main__":
    tf.app.run()