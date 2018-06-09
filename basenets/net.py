import tensorflow as tf
from abc import ABCMeta, abstractmethod
import logging
import six


@six.add_metaclass(ABCMeta)
class Net(object):
    def __init__(self, weight_decay, name='my_net', **kwargs):
        self.name = name
        self.weight_decay = weight_decay
        self.inputs = {}
        self.ground_truth = {}
        self.endpoints = {}
        self.outputs = {}
        self.loss = None

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractmethod
    def get_update_ops(self):
        raise NotImplementedError

    @abstractmethod
    def calc_loss(self):
        logging.warning('Using default loss function!')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outputs['logits'],
                                                              labels=self.ground_truth['labels'])
        loss = tf.reduce_mean(loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        self.loss = loss
        return loss