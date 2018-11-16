import tensorflow as tf
from abc import ABCMeta, abstractmethod
import logging
import six


@six.add_metaclass(ABCMeta)
class Net(object):
    def __init__(self, name='my_net', **kwargs):
        self.name = name
        # self.weight_decay = weight_decay
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
        raise NotImplementedError