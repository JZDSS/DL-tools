from __future__ import print_function
import os
import time

import tensorflow as tf
import tensorflow.contrib.losses as losses
from muti_gpus.nets import build

flags = tf.app.flags

flags.DEFINE_string('data_dir', '../SPCup/patches', 'data direction')
flags.DEFINE_string('log_dir', './logs', 'log direction')
flags.DEFINE_string('ckpt_dir', './ckpt', 'check point direction')
flags.DEFINE_float('weight_decay', 0.00004, 'weight decay')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('max_steps', 64000, 'max steps')
flags.DEFINE_integer('start_step', 0, 'start steps')
flags.DEFINE_string('model_name', 'model', '')
flags.DEFINE_string('gpu', '3', '')
flags.DEFINE_integer('blocks', 5, '')
flags.DEFINE_string('out_file', '', '')
flags.DEFINE_integer('patch_size', 64, '')
flags.DEFINE_string('type', '', '')
flags.DEFINE_integer('num_classes', 10, '')
FLAGS = flags.FLAGS

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
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True
    if not tf.gfile.Exists(FLAGS.data_dir):
        raise RuntimeError('data direction is not exist!')

    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.ckpt_dir):
        tf.gfile.MakeDirs(FLAGS.ckpt_dir)

    f = open(FLAGS.out_file, 'w')
    if not f:
        raise RuntimeError('OUTPUT FILE OPEN ERROR!!!!!!')


    with tf.device('/cpu:0'):
        global_step = tf.Variable(FLAGS.start_step, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, [24000, 48000], [0.1, 0.01, 0.001])
        tf.summary.scalar('learing rate', learning_rate)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum)
        # learning_rate = tf.train.exponential_decay(0.01, global_step, 32000, 0.1)
        # opt = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []
        num_gpus = len(FLAGS.gpu.split(','))
        tower_loss = []
        tower_acc = []
        images, labels = input_pipeline(
            tf.train.match_filenames_once(os.path.join(FLAGS.data_dir, 'train', '*.tfrecords')),
            int(FLAGS.batch_size/num_gpus))
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * num_gpus)
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i) as scope:
                    image_batch, label_batch = batch_queue.dequeue()
                    logits = build.net(image_batch, is_training, FLAGS)

                    losses.sparse_softmax_cross_entropy(logits, label_batch, scope=scope)
                    total_loss = losses.get_losses(scope=scope) + losses.get_regularization_losses(scope=scope)
                    total_loss = tf.add_n(total_loss)

                    grads = opt.compute_gradients(total_loss)
                    tower_grads.append(grads)
                    tower_loss.append(losses.get_losses(scope=scope))

                    with tf.name_scope('accuracy'):
                        correct_prediction = tf.equal(tf.reshape(tf.argmax(logits, 1), [-1, 1]), tf.cast(label_batch, tf.int64))
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    tower_acc.append(accuracy)
                    tf.get_variable_scope().reuse_variables()
        with tf.name_scope('scores'):
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.stack(tower_acc, axis=0))

            with tf.name_scope('batch_loss'):
                batch_loss = tf.add_n(tower_loss)[0]

            tf.summary.scalar('loss', batch_loss)
            tf.summary.scalar('accuracy', accuracy)

        grads = average_gradients(tower_grads)

        # variable_averages = tf.train.ExponentialMovingAverage(
        #     cifar10.MOVING_AVERAGE_DECAY, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
            # train_op = tf.group(apply_gradient_op, variables_averages_op)
            train_op = apply_gradient_op


        # summary_op = tf.summary.merge_all()
        # init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(name="saver")

        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
                saver.restore(sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))
            else:
                sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            train_writer.flush()

            for i in range(FLAGS.start_step, FLAGS.max_steps + 1):
                # feed = feed_dict(True, True)
                if i % 1000 == 0:  # Record summaries and test-set accuracy
                    # loss0 = sess.run([total_loss], feed_dict=feed_dict(False, False))
                    # test_writer.add_summary(summary, i)
                    # feed[is_training] = FLAGS
                    acc, loss, summ = sess.run([accuracy, batch_loss, summary_op], feed_dict={is_training: False})
                    train_writer.add_summary(summ, i)
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=f)
                    # print('step %d: train_acc=%f, train_loss=%f; test_acc=%f, test_loss=%f' % (i, acc1, loss1, acc0, loss0),
                    #       file=f)
                    print('step %d: accuracy=%f, loss=%f' % (i, acc, loss), file=f)
                    saver.save(sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))
                    f.flush()
                sess.run(train_op, feed_dict={is_training: True})

            coord.request_stop()
            coord.join(threads)

    train_writer.close()
    # test_writer.close()
    f.close()


if __name__ == '__main__':
    tf.app.run()

