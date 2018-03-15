import tensorflow as tf
import logging

class Net(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.inputs = None
        self.output = None
        self.ground_truth = None
        self.architecture = []

    def build(self, inputs):
        raise NotImplementedError

    def get_update_ops(self):
        raise NotImplementedError

    def _checkout_loss(self):
        if self.output is None:
            raise ValueError('Can not calculate loss, because attribute "output" is None!')
        if self.ground_truth is None:
            raise ValueError('Can not calculate loss, because attribute "ground_truth" is None!')

    def loss(self, logits, labels, *args, **kwargs):
        logging.warning('Using default loss function!')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return loss