import os
import tensorflow as tf

class Trainer:
    def __init__(self, net, gpus=[0]):
        self.gpus = list(set(gpus))
        self.num_gpus = len(self.gpus)
        self.net = net
        self.loss_fn = self._loss_fn
        self._init_gpus()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def _loss_fn(self, logits, labels, *args, **kwargs):
        # Define loss and add to tf.GraphKeys.LOSSES collection
        # You must add loss to tf.GraphKeys.LOSSES collection as well if you
        # want to redefine loss function. You can use set_loss_fn(your_loss_fn) to
        # enable your custom loss function.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return loss

    def set_loss_fn(self, fn):
        self.loss_fn = fn

    def _average_gradients(self, tower_grads):
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

    def _init_gpus(self):
        # initialize GPU environment
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(n) for n in self.gpus])
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

    def _learning_rate(self, cfg):
        pass
    def _optimizer(self, cfg):
        pass
    def build_graph(self, lr_cfg, opt_cfg):
        with tf.device('/cpu:0'):
            num_gpus = self.num_gpus
            self._learning_rate(lr_cfg)
            opt = self._optimizer(opt_cfg)
            tower_grads = []
            tower_loss = []
            tower_acc = []
            image_batch0 = tf.placeholder(tf.float32, [None, 64, 64, 3], 'imgs')
            label_batch0 = tf.placeholder(tf.int32, [None], 'labels')
            image_batch = tf.split(image_batch0, num_gpus, 0)
            label_batch = tf.split(label_batch0, num_gpus, 0)
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        logits = self.net.build(image_batch[i])
                        with tf.variable_scope('loss'):
                            self.loss_fn(logits=logits, labels=label_batch[i])
                        total_loss = tf.get_collection(tf.GraphKeys.LOSSES, scope=scope) \
                                     + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
                        total_loss = tf.add_n(total_loss)

                        grads = opt.compute_gradients(total_loss)
                        tower_grads.append(grads)
                        tower_loss.append(tf.get_collection(tf.GraphKeys.LOSSES, scope=scope))
                        with tf.name_scope('accuracy'):
                            correct_prediction = tf.equal(tf.reshape(tf.argmax(logits, 1), [-1, 1]),
                                                          tf.cast(label_batch[i], tf.int64))
                            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                        tower_acc.append(accuracy)
                        tf.get_variable_scope().reuse_variables()

            with tf.name_scope('scores'):
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.stack(tower_acc, axis=0))
                with tf.name_scope('batch_loss'):
                    batch_loss = tf.add_n(tower_loss)[0] / num_gpus

                tf.summary.scalar('loss', batch_loss)
                tf.summary.scalar('accuracy', accuracy)

            grads = self._average_gradients(tower_grads)

            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                variable_averages = tf.train.ExponentialMovingAverage(0.9999, self.global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
                train_op = tf.group(apply_gradient_op, variables_averages_op)



    def __del__(self):
        try:
            self.sess.stop()
        except Exception as e:
            print(e)
import cfg
from nets.resnet import *
flags = cfg.FLAGS
def main():
    net = ResNet(flags)
    trainer = Trainer(net)
    trainer.build_graph(None, None)


if __name__ == '__main__':
    main()