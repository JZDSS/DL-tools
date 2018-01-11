from __future__ import print_function
import os
import time

import tensorflow as tf
import tensorflow.contrib.losses as losses
import numpy as np
from multi_gpus.nets import build
import cfg

FLAGS = cfg.FLAGS


is_training = tf.placeholder(tf.bool)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def read_from_tfrecord(tfrecord_file_queue):
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'label': tf.FixedLenFeature([], tf.string),
                                                    'patch_raw': tf.FixedLenFeature([], tf.string)
                                                }, name='features')
    image = tf.decode_raw(tfrecord_features['patch_raw'], tf.uint8)
    ground_truth = tf.decode_raw(tfrecord_features['label'], tf.int32)

    image = tf.cast(tf.reshape(image, [FLAGS.patch_size, FLAGS.patch_size, 3]), tf.float32)
    if FLAGS.aug:
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    ground_truth = tf.reshape(ground_truth, [1])
    return image, ground_truth


def input_pipeline(filenames, batch_size, read_threads=2, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [read_from_tfrecord(filename_queue)
                    for _ in range(read_threads)]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True
    if not tf.gfile.Exists(FLAGS.data_dir):
        raise RuntimeError('data direction is not exist!')

    # if tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # if not tf.gfile.Exists(FLAGS.ckpt_dir):
    tf.gfile.MakeDirs(os.path.join(FLAGS.ckpt_dir, 'best'))

    f = open(FLAGS.out_file + '.txt', 'a' if FLAGS.start_step is not 0 else 'w')
    if not f:
        raise RuntimeError('OUTPUT FILE OPEN ERROR!!!!!!')


    with tf.device('/cpu:0'):
        num_gpus = len(FLAGS.gpu.split(','))
        global_step = tf.Variable(FLAGS.start_step, name='global_step', trainable=False)
        # learning_rate = tf.train.exponential_decay(0.05, global_step, 2000, 0.9, staircase=True)
        learning_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.95, staircase=True)
        # learning_rate = tf.train.piecewise_constant(global_step, [24000, 48000, 72000, 108000, 144000], 
        #                                                 [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])        
        tf.summary.scalar('learing rate', learning_rate)
        # opt = tf.train.AdamOptimizer(learning_rate)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum)
        # opt = tf.train.GradientDescentOptimizer(learning_rate)
        # learning_rate = tf.train.exponential_decay(0.01, global_step, 32000, 0.1)
        # opt = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []
        tower_loss = []
        tower_acc = []
        images_t, labels_t = input_pipeline(
            tf.train.match_filenames_once(os.path.join(FLAGS.data_dir, 'train', '*.tfrecords')), FLAGS.batch_size * num_gpus)
        # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        #     [images, labels], capacity=2 * num_gpus)
        images_v, labels_v = input_pipeline(
            tf.train.match_filenames_once(os.path.join(FLAGS.data_dir, 'valid', '*.tfrecords')), (256 // num_gpus) * num_gpus)
        # batch_queue_v = tf.contrib.slim.prefetch_queue.prefetch_queue(
        #     [images_v, labels_v], capacity=2 * num_gpus)
        image_batch0 = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, 3], 'imgs')
        label_batch0 = tf.placeholder(tf.int32, [None, 1], 'labels')
        image_batch = tf.split(image_batch0, num_gpus, 0)
        label_batch = tf.split(label_batch0, num_gpus, 0)
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i) as scope:
                    logits = build.net(image_batch[i], is_training, FLAGS)
                    losses.sparse_softmax_cross_entropy(labels=label_batch[i], logits=logits, scope=scope)
                    total_loss = losses.get_losses(scope=scope) + losses.get_regularization_losses(scope=scope)
                    total_loss = tf.add_n(total_loss)

                    grads = opt.compute_gradients(total_loss)
                    tower_grads.append(grads)
                    tower_loss.append(losses.get_losses(scope=scope))

                    with tf.name_scope('accuracy'):
                        correct_prediction = tf.equal(tf.reshape(tf.argmax(logits, 1), [-1, 1]), tf.cast(label_batch[i], tf.int64))
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    tower_acc.append(accuracy)
                    tf.get_variable_scope().reuse_variables()

        with tf.name_scope('scores'):
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.stack(tower_acc, axis=0))
            with tf.name_scope('batch_loss'):
                batch_loss = tf.add_n(tower_loss)[0]/num_gpus

            tf.summary.scalar('loss', batch_loss)
            tf.summary.scalar('accuracy', accuracy)

        grads = average_gradients(tower_grads)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
            train_op = tf.group(apply_gradient_op, variables_averages_op)
            # train_op = apply_gradient_op


        # summary_op = tf.summary.merge_all()
        # init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(name="saver", max_to_keep=10)
        saver_best = tf.train.Saver(name='best', max_to_keep=100)
        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            else:
                sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            train_writer.flush()
            valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid', sess.graph)
            valid_writer.flush()
            cache = np.ones(5, dtype=np.float32)/FLAGS.num_classes
            cache_v = np.ones(5, dtype=np.float32)/FLAGS.num_classes
            d = 1000
            best = 0
            for i in range(FLAGS.start_step, FLAGS.max_steps + 1):
                def get_batch(set, on_training):
                    if set == 'train':
                        img, lb = sess.run([images_t, labels_t])
                    elif set == 'valid':
                        img, lb = sess.run([images_v, labels_v])
                    else:
                        raise RuntimeError('Unknown set name')
                    feed_dict = {}
                    feed_dict[image_batch0] = img
                    feed_dict[label_batch0] = lb
                    feed_dict[is_training] = on_training
                    return feed_dict
                # feed = feed_dict(True, True)
                if i % d == 0:  # Record summaries and test-set accuracy
                    # loss0 = sess.run([total_loss], feed_dict=feed_dict(False, False))
                    # test_writer.add_summary(summary, i)
                    # feed[is_training] = FLAGS
                    acc, loss, summ, lr = sess.run([accuracy, batch_loss, summary_op, learning_rate], feed_dict=get_batch('train', False))
                    cache[int(i/d)%5] = acc
                    acc_v, loss_v, summ_v = sess.run([accuracy, batch_loss, summary_op], feed_dict=get_batch('valid', False))
                    cache_v[int(i/d)%5] = acc_v
                    train_writer.add_summary(summ, i)
                    valid_writer.add_summary(summ_v, i)
                    print(('step %d, ' % i) + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=f)
                    print('acc(t)=%f(%f), loss(t)=%f;\nacc(v)=%f(%f), loss(v)=%f; lr=%e' % (acc, cache.mean(), loss, acc_v, cache_v.mean(), loss_v, lr), file=f)
                    saver.save(sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name), global_step=i)
                    if acc_v > 0.90:
                        saver_best.save(sess, os.path.join(FLAGS.ckpt_dir, 'best', FLAGS.model_name), global_step=i)
                    f.flush()
                sess.run(train_op, feed_dict=get_batch('train', True))

            coord.request_stop()
            coord.join(threads)

    train_writer.close()
    # test_writer.close()
    f.close()


if __name__ == '__main__':
    tf.app.run()

