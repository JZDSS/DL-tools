import six
from abc import ABCMeta, abstractmethod
from io.reader import Reader
import tensorflow as tf


@six.add_metaclass(ABCMeta)
class TFRecordsReader(Reader):

    def __init__(self, batch_size, fake=False, **kwargs):
        super(TFRecordsReader, self).__init__(batch_size, fake, **kwargs)

    @abstractmethod
    def _parser(self, record):
        raise NotImplementedError

    def _get_iterator(self, filenames, num_readers, read_threads, num_epochs):
        dataset = tf.data.Dataset.list_files(filenames, True)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset, num_readers, 1)
        dataset = dataset.map(self._parser, read_threads)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        # dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def read(self, filenames, num_readers, read_threads, num_epochs=None):
        iterator = None
        if not self.fake:
            iterator = self._get_iterator(filenames, num_readers, read_threads, num_epochs)
        return self._post_process(iterator)

    @abstractmethod
    def _post_process(self, iterator):
        raise NotImplementedError
