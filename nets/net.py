import tensorflow as tf
import logging

class Net(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.inputs = {}
        self.ground_truth = {}
        self.endpoints = {}

    def build(self, inputs):
        raise NotImplementedError

    def get_update_ops(self):
        raise NotImplementedError

    def _checkout_loss(self):
        pass

    def loss(self, logits, labels, *args, **kwargs):
        logging.warning('Using default loss function!')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return loss