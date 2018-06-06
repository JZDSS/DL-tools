import six
from abc import ABCMeta, abstractmethod
from inputs.inputs import Inputs

@six.add_metaclass(ABCMeta)
class TFRecordsInputs(Inputs):

    def __init__(self):
        super(TFRecordsInputs, self).__init__()

    @abstractmethod
    def _read_from_tfrecord(self, tfrecord_file_queue):
        raise NotImplementedError
