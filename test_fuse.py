from __future__ import print_function
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.patch import get_patches
from muti_gpus.nets import build

flags = tf.app.flags

flags.DEFINE_string('data_dir', '../data', 'data direction')
flags.DEFINE_string('ckpt_dir', './ckpt', 'check point direction')
flags.DEFINE_integer('patches', 128, 'batch size')
flags.DEFINE_string('model_name', 'model', '')
flags.DEFINE_integer('patch_size', 64, '')
flags.DEFINE_string('set', 'valid', '')
flags.DEFINE_string('meta_dir', './meta', '')
flags.DEFINE_string('gpu', '3', '')
flags.DEFINE_integer('blocks', 3, '')
flags.DEFINE_string('out_file', '', '')
flags.DEFINE_string('type', '', '')
flags.DEFINE_integer('num_classes', 10, '')
flags.DEFINE_float('weight_decay', 0.00004, '')

FLAGS = flags.FLAGS


def standardization(x):
    mean = np.mean(x)
    stddev = np.std(x)
    adjusted_stddev = max(stddev, 1./np.sqrt(FLAGS.patch_size * FLAGS.patch_size * 3))
    return (x - mean) / adjusted_stddev


def main(_):
    ff = open(FLAGS.out_file, 'w')
    if not ff:
        raise RuntimeError('OUTPUT FILE OPEN ERROR!!!!!!')

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    if not tf.gfile.Exists(FLAGS.data_dir):
        raise RuntimeError('data direction is not exist!')

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, 3], 'x')

    with tf.variable_scope('compress70'):
        y1 = build.net(x, False, FLAGS)
    varlist1 = {v.op.name.replace('compress70/', ''): v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="compress70")}
    saver1 = tf.train.Saver(var_list=varlist1)

    with tf.variable_scope('compress90'):
        y2 = build.net(x, False, FLAGS)
    varlist2 = {v.op.name.replace('compress90/', ''): v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="compress90/")}
    saver2 = tf.train.Saver(var_list=varlist2)

    with tf.variable_scope('gamma0_8'):
        y3 = build.net(x, False, FLAGS)
    varlist3 = {v.op.name.replace('gamma0_8/', ''): v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="gamma0_8/")}
    saver3 = tf.train.Saver(var_list=varlist3)

    with tf.variable_scope('gamma1_2'):
        y4 = build.net(x, False, FLAGS)
    varlist4 = {v.op.name.replace('gamma1_2/', ''): v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="gamma1_2/")}
    saver4 = tf.train.Saver(var_list=varlist4)

    with tf.variable_scope('resize0_5'):
        y5 = build.net(x, False, FLAGS)
    varlist5 = {v.op.name.replace('resize0_5/', ''): v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="resize0_5/")}
    saver5 = tf.train.Saver(var_list=varlist5)

    with tf.variable_scope('resize0_8'):
        y6 = build.net(x, False, FLAGS)
    varlist6 = {v.op.name.replace('resize0_8/', ''): v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="resize0_8/")}
    saver6 = tf.train.Saver(var_list=varlist6)

    with tf.variable_scope('resize1_5'):
        y7 = build.net(x, False, FLAGS)
    varlist7 = {v.op.name.replace('resize1_5/', ''): v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="resize1_5/")}
    saver7 = tf.train.Saver(var_list=varlist7)

    with tf.variable_scope('resize2_0'):
        y8 = build.net(x, False, FLAGS)
    varlist8 = {v.op.name.replace('resize2_0/', ''): v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="resize2_0/")}
    saver8 = tf.train.Saver(var_list=varlist8)

    y = tf.add_n([y1, y2, y3, y4, y5, y6, y7, y8])
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    pred = tf.nn.softmax(y, 1)

    f = open(os.path.join(FLAGS.meta_dir, FLAGS.set) + '.txt', 'r')
    image_names = []
    labels = []
    line = f.readline()
    while line:
        l = line.split(' ')
        if len(l) == 2:
            image_name = l[0]
            label = l[1]
        else:
            image_name = l[0] + ' ' + l[1]
            label = l[2]
        # image_name, label = line.split(' ')
        label = label[0:-1]
        image_names.append(image_name)
        labels.append(int(label))
        line = f.readline()
    f.close()

    f = open(os.path.join(FLAGS.meta_dir, 'spc_classes.txt'), 'r')
    meta = {}
    line = f.readline()
    while line:
        label, class_name = line.split(' ')
        class_name = class_name[0:-1]
        meta[int(label)] = class_name
        line = f.readline()
    f.close()
    confusion = np.zeros(shape=(10, 10), dtype=np.uint32)
    confusion_i = np.zeros(shape=(10, 10), dtype=np.uint32)
    total = 0.
    correct = 0.
    total_p = 0.
    correct_p = 0.
    with tf.Session(config=config) as sess:
        # if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
        #     saver.restore(sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))
        # else:
        #     raise RuntimeError("Check point files don't exist!")
        saver1.restore(sess, os.path.join('stand_alone', 'compress70/ckpt/model'))
        saver2.restore(sess, os.path.join('stand_alone', 'compress90/ckpt/model'))
        saver3.restore(sess, os.path.join('stand_alone', 'gamma0_8/ckpt/model'))
        saver4.restore(sess, os.path.join('stand_alone', 'gamma1_2/ckpt/model'))
        saver5.restore(sess, os.path.join('stand_alone', 'resize0_5/ckpt/model'))
        saver6.restore(sess, os.path.join('stand_alone', 'resize0_8/ckpt/model'))
        saver7.restore(sess, os.path.join('stand_alone', 'resize1_5/ckpt/model'))
        saver8.restore(sess, os.path.join('stand_alone', 'resize2_0/ckpt/model'))

        for i in range(len(labels)):
            label = labels[i]
            class_name = meta[label]
            image_name = image_names[i]
            full_path = os.path.join(FLAGS.data_dir, class_name, image_name)
            img = plt.imread(full_path)
            data = np.ndarray(shape=(FLAGS.patches, FLAGS.patch_size, FLAGS.patch_size, 3), dtype=np.float32)
            for n, patch in enumerate(get_patches(img, FLAGS.patches, FLAGS.patch_size)):
                patch = standardization(patch)
                data[n, :] = patch
            # data = standardization(data)
            prediction = sess.run(pred, feed_dict={x: data})
            prediction0 = np.argmax(prediction, 1)
            for n in prediction0:
                if n == label:
                    correct_p = correct_p + 1
                confusion[label, n] = confusion[label, n] + 1
            total_p = total_p + FLAGS.patches
            # count = np.bincount(prediction)
            # prediction = np.argmax(count)
            prediction = np.sum(prediction, 0)
            #print(prediction)
            prediction = np.argmax(prediction)
            confusion_i[label, prediction] = confusion_i[label, prediction] + 1
            print("predict %d while true label is %d." % (prediction, label), file=ff)
            ff.flush()
            total = total + 1
            if prediction == label:
                correct = correct + 1
    print('accuracy(patch level) = %f' % (correct_p / total_p), file=ff)
    print('accuracy(image level) = %f' % (correct / total), file=ff)
    print('confusion matrix--patch level:', file=ff)
    print(confusion, file=ff)
    print('confusion matrix--image level:', file=ff)
    print(confusion_i, file=ff)
    print('/|\\', file=ff)
    print(' |', file=ff)
    print('actual', file=ff)
    print(' |', file=ff)
    print(' ---prediction--->', file=ff)
    ff.close()

if __name__ == '__main__':
    tf.app.run()

