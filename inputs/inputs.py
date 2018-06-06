from abc import ABCMeta, abstractmethod
import six

@six.add_metaclass(ABCMeta)
class Inputs(object):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def input_pipeline(self, filenames, batch_size, read_threads, num_epochs=None):
        raise NotImplementedError
