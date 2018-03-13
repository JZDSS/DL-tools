import os
import tensorflow as tf
from ssd_cfg import FLAGS as FLAGS
from nets import vgg16_ssd, ssd_input


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


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True
    if not tf.gfile.Exists(FLAGS.data_dir):
        raise RuntimeError('data direction is not exist!')

    tf.gfile.MakeDirs(FLAGS.log_dir)

    f = open(FLAGS.out_file + '.txt', 'a' if FLAGS.start_step is not 0 else 'w')
    if not f:
        raise RuntimeError('OUTPUT FILE OPEN ERROR!!!!!!')


    with tf.device('/cpu:0'):
        num_gpus = len(FLAGS.gpu.split(','))
        global_step = tf.Variable(FLAGS.start_step, name='global_step', trainable=False)

        # learning_rate = tf.train.piecewise_constant(global_step,
        #                                             [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
        #                                             [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
        # step_size = 10000
        # learning_rate = tf.train.exponential_decay(1.0, global_step, 2*step_size, 0.5, staircase=True)
        # cycle = tf.floor(1 + tf.cast(global_step, tf.float32) / step_size / 2.)
        # xx = tf.abs(tf.cast(global_step, tf.float32)/step_size - 2. * tf.cast(cycle, tf.float32) + 1.)
        # learning_rate = 1e-4 + (1e-1 - 1e-4) * tf.maximum(0., (1-xx))*learning_rate
        # learning_rate = tf.train.piecewise_constant(global_step, [10000, 70000, 120000, 170000, 220000],
        #                                                         [0.01, 0.1, 0.001, 0.0001, 0.00001, 0.000001])
        # learning_rate = tf.constant(0.001)
        learning_rate = tf.train.exponential_decay(0.001, global_step, 30000, 0.1, staircase=True)
        print('learning_rate = tf.train.exponential_decay(0.05, global_step, 30000, 0.1, staircase=True)', file=f)

        # opt = tf.train.AdamOptimizer(learning_rate)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        # opt = tf.train.GradientDescentOptimizer(learning_rate)
        # learning_rate = tf.train.exponential_decay(0.01, global_step, 32000, 0.1)
        # opt = tf.train.GradientDescentOptimizer(learning_rate)
        print('opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)', file=f)
        print('weight decay = %e' % FLAGS.weight_decay, file=f)
        f.flush()
        tf.summary.scalar('learing rate', learning_rate)
        tower_grads = []
        tower_loss = []

        i_and_l = ssd_input.input_pipeline(
            tf.train.match_filenames_once(os.path.join('nets/ssd', '*.tfrecords')),
            FLAGS.batch_size * num_gpus, read_threads=1)
        images = i_and_l[0]
        locations = i_and_l[1:len(i_and_l) // 2 + 1]
        labels = i_and_l[len(i_and_l) // 2 + 1:]

        net = vgg16_ssd.vgg16_ssd()

        image_batch = tf.split(images, num_gpus, 0)
        tmp = []
        for loc in locations:
            tmp.append(tf.split(loc, num_gpus, 0))
        location_batch = []
        for i in range(num_gpus):
            location_batch.append(list(m[i] for m in tmp))

        tmp = []
        for lab in labels:
           tmp.append(tf.split(lab, num_gpus, 0))
        label_batch = []
        for i in range(num_gpus):
            label_batch.append(list(m[i] for m in tmp))

        with tf.name_scope('CPU'):
            net.build(tf.placeholder(dtype=tf.float32, shape=[None, 300, 300, 3]))

    for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    net.build(image_batch[i])
                    loss = net.get_loss(location_batch[i], label_batch[i])
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
                    tower_loss.append(loss)


        with tf.name_scope('scores'):
            with tf.name_scope('batch_loss'):
                batch_loss = tf.add_n(tower_loss)/num_gpus

            tf.summary.scalar('loss', batch_loss)

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
        with tf.Session(config=config) as sess:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
                try:
                    print('restore from ckpt')
                    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
                except:
                    print('restore failed, load weights from vgg16')
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
            else:
                print('train from vgg16')
                sess.run(tf.global_variables_initializer())
                sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
            if FLAGS.start_step != 0:
                sess.run(tf.assign(global_step, FLAGS.start_step))
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            train_writer.flush()
            d = 1000
            for i in range(FLAGS.start_step, FLAGS.max_steps + 1):
                # feed = feed_dict(True, True)
                if i % d == 0:  # Record summaries and test-set accuracy
                    loss, summ, lr = sess.run([batch_loss, summary_op, learning_rate])
                    train_writer.add_summary(summ, i)
                    saver.save(sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name), global_step=i)
                    f.flush()
                sess.run(train_op)

            coord.request_stop()
            coord.join(threads)

    train_writer.close()
    f.close()


if __name__ == '__main__':
    tf.app.run()

