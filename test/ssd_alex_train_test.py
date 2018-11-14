import tensorflow as tf
import os
from ssd import configure
from inputs.ssdinputs import SSDInputs
from builder.model_builder import ModelBuilder
import time
# tf.enable_eager_execution()

def main(_):
    config = configure.Configure('../ssd/ssd_alex.config')
    config = config.get_config()

    log_dir = config['train']['log_dir']
    ckpt_dir = config['train']['ckpt_dir']

    builder = ModelBuilder(config, fake=False, mode='train', input_class=SSDInputs)
    net = builder.model
    loss = net.calc_loss()

    optimizer = tf.train.AdamOptimizer(0.0005, 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    saver = tf.train.Saver(name="saver", max_to_keep=5)
    summ = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        # o_l, o_c, gt_l, gt_c = sess.run([net.outputs['location'],
        #                                  net.outputs['classification'],
        #                                  net.ground_truth['locations'],
        #                                  net.ground_truth['labels']], feed_dict={net.is_training: True})
        # c = sess.run(tf.get_default_graph().get_tensor_by_name('pred_0/weights:0'))
        # b = sess.run(net.inputs)
        # a = sess.run(net.endpoints, feed_dict={net.is_training: True})
        i = 0
        if os.listdir(ckpt_dir):
            print('Train from ckpt')
            dir = tf.train.latest_checkpoint(ckpt_dir)
            i = int(dir.split('-')[-1])
            saver.restore(sess, dir)
        else:
            print('Try to train from pre-trained net')
            init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)
            try:
                assert init_ops
                sess.run(init_ops)
            except Exception as e:
                print(e)
                print('Failed! Train from 0')
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        train_writer.flush()
        
        d = 100

        while True:
            try:
                if i % d == 0:
                    saver.save(sess, os.path.join(ckpt_dir, 'ssd_model'), global_step=i)
                    summary = sess.run(summ, feed_dict={net.is_training: False})
                    train_writer.add_summary(summary, i)
                    train_writer.flush()
                    if i != 0:
                        time.sleep(10)
                sess.run(train_op, feed_dict={net.is_training: True})
                i += 1
            except tf.errors.OutOfRangeError as e:
                print(e)
                saver.save(sess, os.path.join(ckpt_dir, 'ssd_model'), global_step=i - 1)
                break

if __name__ == "__main__":
    tf.app.run()