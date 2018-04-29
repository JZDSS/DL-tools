from __future__ import print_function
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.patch import get_patches
from multi_gpus.nets import build
import tensorflow as tf
from multi_gpus import layers

def block(inputs, num_outputs, weight_decay, scope, is_training, down_sample = False):
    with tf.variable_scope(scope):
        num_inputs = inputs.get_shape().as_list()[3]

        res = layers.conv(inputs, num_outputs = num_outputs, kernel_size=[3, 3],
                          strides=[1, 2, 2, 1] if down_sample else [1, 1, 1, 1],
                          scope='conv1', b_norm=True, is_training=is_training, weight_decay=weight_decay)

        res = layers.conv(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                          scope='conv2', b_norm=True, is_training=is_training, weight_decay=weight_decay)
        if  num_inputs != num_outputs:
            inputs = layers.conv(inputs, num_outputs=num_outputs, kernel_size=[1, 1], activation_fn=None,
                                 scope='short_cut', strides=[1, 2, 2,1 ], b_norm=True, is_training=is_training,
                                 weight_decay=weight_decay)
        res = tf.nn.relu(res + inputs)

    return res


def build_single_net(x, is_training, FLAGS):
    n = FLAGS.blocks
    # shape = x.get_shape().as_list()
    with tf.variable_scope('pre'):
        pre = layers.conv(x, num_outputs=16,  kernel_size = [3, 3], scope='conv', b_norm=True, is_training=is_training,
                          weight_decay=FLAGS.weight_decay)
        # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
    h = pre
    for i in range(1, n + 1):
        h = block(h, 16, FLAGS.weight_decay, '16_block{}'.format(i), is_training)

    h = block(h, 32, FLAGS.weight_decay, '32_block1', is_training, True)
    for i in range(2, n + 1):
        h = block(h, 32, FLAGS.weight_decay, '32_block{}'.format(i), is_training)

    h = block(h, 64, FLAGS.weight_decay, '64_block1', is_training, True)
    for i in range(2, n + 1):
        h = block(h, 64, FLAGS.weight_decay, '64_block{}'.format(i), is_training)

    shape = h.get_shape().as_list()

    h = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool')
    # res = layers.conv(h, num_outputs=FLAGS.num_classes, kernel_size=[1, 1], scope='fc1', padding='VALID',
    #                 b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)

    return h



def build_net(x, is_training, FLAGS):
    num_branches = FLAGS.num_branches
    y = []
    for i in range(1, num_branches + 1):
        with tf.variable_scope('branch%d' % i):
            tmp = build_single_net(x, is_training, FLAGS)
            if FLAGS.p_relu:
                tmp = layers.p_relu(tmp)
        y.append(tmp)

    with tf.variable_scope('branch%d' % (num_branches + 1)):
        y4 = build_single_net(x, is_training, FLAGS)
        
    con = tf.concat(y, axis=3) #192
    con = tf.reshape(con, [-1, 1, num_branches*64])

    w = layers.conv(y4, num_outputs=FLAGS.num_classes*num_branches*64, kernel_size=[1, 1], scope='fc2', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)
    w = tf.reshape(w, [-1, num_branches*64, FLAGS.num_classes])

    res = tf.matmul(con, w)
    res = layers.batch_norm(res, is_training=is_training, scope='bn')
    return w, tf.reshape(res, [-1, FLAGS.num_classes])



flags = tf.app.flags

flags.DEFINE_string('ckpt_dir', './ckpt', 'check point direction')
flags.DEFINE_integer('patches', 1, 'batch size')
flags.DEFINE_string('model_name', None, '')
flags.DEFINE_integer('patch_size', 64, '')
flags.DEFINE_string('set', 'valid', '')
flags.DEFINE_string('meta_dir', './meta', '')
flags.DEFINE_string('gpu', '3', '')
flags.DEFINE_integer('blocks', 3, '')
flags.DEFINE_string('out_file', '', '')
flags.DEFINE_string('type', '', '')
flags.DEFINE_integer('num_classes', 10, '')
flags.DEFINE_float('weight_decay', 0.00004, '')
flags.DEFINE_string('extra', '', 'j70, j90, g0_8 etc')
flags.DEFINE_string('format', '', 'jpg or tif')
flags.DEFINE_integer('num_branches', 0, '')
flags.DEFINE_boolean('p_relu', False, '')
FLAGS = flags.FLAGS

# def get_patches(img, max_patches, patch_size):
#     h = img.shape[0]
#     w = img.shape[1]
#     n = 0
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     for start_r in range(0, h - patch_size + 1):
#         for  start_c in range(0, w - patch_size + 1):
#             patch = img[start_r:start_r + patch_size, start_c:start_c + patch_size, :]
#             n = n + 1
#             yield patch


def standardization(x):
    mean = np.mean(x)
    stddev = np.std(x)
    adjusted_stddev = max(stddev, 1./np.sqrt(FLAGS.patch_size * FLAGS.patch_size * 3))
    return (x - mean) / adjusted_stddev


def main(_):
    # ff = open(FLAGS.out_file, 'w')
    # if not ff:
    #     raise RuntimeError('OUTPUT FILE OPEN ERROR!!!!!!')
    # print('fname,camera', file=ff)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, 3], 'x')

    w, _ = build_net(x, False, FLAGS)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    # pred = tf.nn.softmax(y, 1)

    with tf.name_scope("saver"):
        saver = tf.train.Saver(name="saver")

    # f = open(os.path.join(FLAGS.meta_dir, FLAGS.set) + '.txt', 'r')
    # image_names = []
    # labels = []
    # line = f.readline()
    # while line:
    #     l = line.split(' ')
    #     if len(l) == 2:
    #         image_name = l[0]
    #         label = l[1]
    #     else:
    #         image_name = l[0] + ' ' + l[1]
    #         label = l[2]
    #     # image_name, label = line.split(' ')
    #     label = label[0:-1]
    #     image_names.append(image_name.split('.')[0] + '-' + FLAGS.extra + '.' + FLAGS.format)
    #     labels.append(int(label))
    #     line = f.readline()
    # f.close()
    image_names = os.listdir('/data/spcup_test')

    # f = open(os.path.join(FLAGS.meta_dir, 'spc_classes.txt'), 'r')
    # meta = {}
    # line = f.readline()
    # while line:
    #     label, class_name = line.split(' ')
    #     class_name = class_name[0:-1]
    #     meta[int(label)] = class_name
    #     line = f.readline()
    # f.close()
    # confusion = np.zeros(shape=(10, 10), dtype=np.uint32)
    # confusion_i = np.zeros(shape=(10, 10), dtype=np.uint32)
    # total = 0.
    # correct = 0.
    # total_p = 0.
    # correct_p = 0.
    with tf.Session(config = config) as sess:
        if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir) if FLAGS.model_name is None else os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))
        else:
            raise RuntimeError("Check point files don't exist!")

        for i in range(len(image_names)):
            # label = labels[i]
            # class_name = meta[label]
            image_name = image_names[i]
            print(image_name)
            full_path = os.path.join('/data/spcup_test', image_name)
            img = plt.imread(full_path)
            
            data = np.ndarray(shape=(FLAGS.patches, FLAGS.patch_size, FLAGS.patch_size, 3), dtype=np.float32)
            for n, patch in enumerate(get_patches(img, FLAGS.patches, FLAGS.patch_size)):
                patch = standardization(patch)
                data[n, :] = patch
            # data = standardization(data)
            ww = sess.run(w, feed_dict={x: data})[0, :]
            # print(ww.shape)
            plt.imshow(ww)
            plt.show()
            # prediction = np.argmax(prediction, 1)
            # for n in prediction0:
            #     if n == label:
            #         correct_p = correct_p + 1
            #     confusion[label, n] = confusion[label, n] + 1
            # total_p = total_p + FLAGS.patches
            # count = np.bincount(prediction)
            # prediction = np.argmax(count)
            # prediction = np.sum(prediction, 0)
            #print(prediction)
            # prediction = np.argmax(prediction)
            
            # print("%s,%s" % (image_name, meta[prediction]), file=ff)
            # ff.flush()
            
    # ff.close()

if __name__ == '__main__':
    tf.app.run()

