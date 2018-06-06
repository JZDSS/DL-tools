import tensorflow as tf
import os
from ssd import configure
from ssd.ssd_input import SSDInput
from model_builder import ModelBuilder
import time


def main(_):
    config = configure.Configure('/home/yqi/Desktop/workspace/PycharmProjects/DL-tools/ssd/ssdd.config')
    config = config.get_config()
    log_dir = config['train']['log_dir']
    ckpt_dir = config['train']['ckpt_dir']
    builder = ModelBuilder(config, fake=False)
    builder(input_class=SSDInput, anchor_config=config['anchor'])
    net = builder.model
    loss = net.loss()

    optimizer = tf.train.AdamOptimizer(0.001)
    train_op = optimizer.minimize(loss)
    saver = tf.train.Saver(name="saver", max_to_keep=10)
    summ = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        i = 0
        if os.listdir(ckpt_dir):
            print('train from ckpt')
            dir = tf.train.latest_checkpoint(ckpt_dir)
            i = int(dir.split('-')[-1])
            saver.restore(sess, dir)
        else:
            print('train from alexnet')
            sess.run(tf.get_collection(tf.GraphKeys.INIT_OP))
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        train_writer.flush()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        d = 100

        while True:
            if i % d == 0:
                saver.save(sess, os.path.join(ckpt_dir, 'ssd_model'), global_step=i)
                summary = sess.run(summ)
                train_writer.add_summary(summary, i)
                train_writer.flush()
                if i != 0:
                    time.sleep(10)
            sess.run(train_op)
            i += 1

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()