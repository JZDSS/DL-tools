from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class Inputs(object):
    
    def __init__(self, batch_size, fake, **kwargs):
        self.batch_size = batch_size
        self.fake = fake
    
    @abstractmethod
    def input_pipeline(self, filenames, num_readers, read_threads, num_epochs=None):
        net_inputs = {}
        ground_truth = {}
        return net_inputs, ground_truth
