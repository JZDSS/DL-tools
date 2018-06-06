from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class SSDBase(object):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def base_net(self):
        raise NotImplementedError

    @abstractmethod
    def down_sample(self):
        raise NotImplementedError

    @abstractmethod
    def extra_net(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError
