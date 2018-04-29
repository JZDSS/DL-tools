# coding:utf-8
from __future__ import print_function
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
from utils.patch import get_patches
import time


flags = tf.app.flags

flags.DEFINE_string('data_dir', '', 'Data direction')
flags.DEFINE_string('out_dir', '', 'Output direction')
flags.DEFINE_integer('patch_size', 64, '')
flags.DEFINE_integer('max_patches', 100, 'number of patches that one image can generate at most')
flags.DEFINE_string('meta_dir', '', '')
flags.DEFINE_string('out_file', '', '')
flags.DEFINE_integer('all_label', None, '')
flags.DEFINE_string('extra', '', 'j70, j90, g0_8 etc')
flags.DEFINE_string('format', '', 'jpg or tif')

FLAGS = flags.FLAGS


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):

    tf.gfile.MakeDirs(os.path.join(FLAGS.out_dir, 'train'))
    tf.gfile.MakeDirs(os.path.join(FLAGS.out_dir, 'valid'))

    temp_name = '%.06f' % time.time()
    if not tf.gfile.Exists(temp_name):
        tf.gfile.MakeDirs(os.path.join(temp_name, 'train'))
        tf.gfile.MakeDirs(os.path.join(temp_name, 'valid'))

    ff = open(FLAGS.out_file, 'w')
    if not ff:
        raise RuntimeError('OUTPUT FILE OPEN ERROR!!!!!!')

    train_name = os.path.join(FLAGS.out_dir, 'train', 'spc_train.tfrecords')
    valid_name = os.path.join(FLAGS.out_dir, 'valid', 'spc_valid.tfrecords')

    train_writer = tf.python_io.TFRecordWriter(train_name)
    valid_writer = tf.python_io.TFRecordWriter(valid_name)

    if not tf.gfile.Exists(FLAGS.meta_dir):
        tf.gfile.MakeDirs(FLAGS.meta_dir)
        classes = os.listdir(FLAGS.data_dir)
        spc_classes = open(os.path.join(FLAGS.meta_dir, 'spc_classes.txt'), 'w')
        train_list = open(os.path.join(FLAGS.meta_dir, 'train.txt'), 'w')
        valid_list = open(os.path.join(FLAGS.meta_dir, 'valid.txt'), 'w')
        for label, class_name in enumerate(classes):
            spc_classes.write(('%d ' % label) + class_name + '\n')
            spc_classes.flush()
            img_names = os.listdir(os.path.join(FLAGS.data_dir, class_name))
            for img_name in img_names:
                full_path = os.path.join(FLAGS.data_dir, class_name, img_name)
                print('processing ' + full_path, file=ff)
                ff.flush()
                img = plt.imread(full_path)
                dice = np.random.randint(0, 5, 1)
                dd = dice
                # writer = train_writer if dice != 0 else valid_writer
                if dd != 0:
                    sett = 'train'
                    train_list.write(img_name.split('-')[0] + (' %d\n' % label))
                else:
                    sett = 'valid'
                    valid_list.write(img_name.split('-')[0] + (' %d\n' % label))
                n = 0
                for patch in get_patches(img, FLAGS.max_patches, FLAGS.patch_size):
                    n = n + 1
                    np.save(os.path.join(temp_name, sett, class_name + '_' + img_name + '_' + str(n)) + '.npy', {'label': label, 'patch': patch})
        spc_classes.close()
        train_list.close()
        valid_list.close()
    else:
        f = open(os.path.join(FLAGS.meta_dir, 'spc_classes.txt'), 'r')
        meta = {}
        line = f.readline()
        while line:
            label, class_name = line.split(' ')
            class_name = class_name[0:-1]
            meta[int(label)] = class_name
            line = f.readline()
        f.close()
        def save_npy(sett):
            f = open(os.path.join(FLAGS.meta_dir, sett) + '.txt', 'r')
            image_names = []
            labels = []
            line = f.readline()
            while line:
                image_name, label = line.split(' ')
                label = label[0:-1]
                image_names.append(image_name + '-' + FLAGS.extra + '.' + FLAGS.format)
                labels.append(int(label))
                line = f.readline()
            f.close()
            for i, img_name in enumerate(image_names):
                full_path = os.path.join(FLAGS.data_dir, meta[labels[i]], img_name)
                print('processing ' + full_path, file=ff)
                ff.flush()
                img = plt.imread(full_path)
                n = 0
                for patch in get_patches(img, FLAGS.max_patches, FLAGS.patch_size):
                    n = n + 1
                    np.save(os.path.join(temp_name, sett, meta[labels[i]] + '_' + img_name + '_' + str(n)) + '.npy',
                            {'label': labels[i], 'patch': patch})

        save_npy('train')
        save_npy('valid')

    train = os.listdir(os.path.join(temp_name, 'train'))
    valid = os.listdir(os.path.join(temp_name, 'valid'))
    # print(len(train))
    # print(len(valid))
    idx = list(range(len(train)))
    shuffle(idx)
    for i in idx:
        dic = np.load(os.path.join(temp_name, 'train', train[i])).item()
        patch = dic['patch']
        label = dic['label']
        if FLAGS.all_label is not None:
            pre = FLAGS.all_label
        patch_raw = patch.tostring()
        label_raw = np.array([label]).astype(np.int32).tostring()
        pre_raw = np.array([pre]).astype(np.int32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'patch_raw': _bytes_feature(patch_raw),
            'label': _bytes_feature(label_raw),
            'pre': _bytes_feature(pre_raw)}))
        train_writer.write(example.SerializeToString())

    idx = list(range(len(valid)))
    shuffle(idx)
    for i in idx:
        dic = np.load(os.path.join(temp_name, 'valid', valid[i])).item()
        patch = dic['patch']
        label = dic['label']
        if FLAGS.all_label is not None:
            pre = FLAGS.all_label
        patch_raw = patch.tostring()
        label_raw = np.array([label]).astype(np.int32).tostring()
        pre_raw = np.array([pre]).astype(np.int32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'patch_raw': _bytes_feature(patch_raw),
            'label': _bytes_feature(label_raw),
            'pre': _bytes_feature(pre_raw)}))
        valid_writer.write(example.SerializeToString())

    train_writer.close()
    valid_writer.close()

    ff.close()
    tf.gfile.DeleteRecursively(temp_name)


if __name__ == "__main__":
    tf.app.run()