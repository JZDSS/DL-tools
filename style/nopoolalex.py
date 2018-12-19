import tensorflow as tf
from basenets import alexnet
import numpy as np
import os
import pickle
from oi.tfrecordsreader import TFRecordsReader
tf.enable_eager_execution()

class Converter(object):

    def __init__(self, data_dir, out_dir):
        self.data_dir = data_dir
        self.out_dir = out_dir

    def _tolist(self, value):
        if not isinstance(value, list):
            value = [value]
        return value

    def _bytes_feature(self, value):
        value = self._tolist(value)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _int64_feature(self, value):
        value = self._tolist(value)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(self, value):
        value = self._tolist(value)
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def write(self):
        raise NotImplementedError


class CifarConverter(Converter):
    def __init__(self, data_dir, out_dir):
        super(CifarConverter, self).__init__(data_dir, out_dir)

    def _unpickle(self, filename):
        # 打开文件
        with open(filename, 'rb') as fo:
            # Python 版的 CIFAR-10 数据集是用 Pickle 进行编码保存的，因此这里用 Pickle 对文件进行解码，得到图片的像素值和标注
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    def __load_train_data(self, data_dir):

        data = np.ndarray(shape=(0, 32 * 32 * 3), dtype=np.uint8)
        labels = np.ndarray(shape=0, dtype=np.int64)

        for i in range(5):
            filename = os.path.join(data_dir, "data_batch_{}".format(i + 1))
            tmp = self._unpickle(filename)

            data = np.append(data, tmp[b'data'], axis=0)
            labels = np.append(labels, tmp[b'labels'], axis=0)
            print('load training data: data_batch_{}'.format(i + 1))

        data = np.reshape(data, [-1, 32, 32, 3], 'F').transpose((0, 2, 1, 3))
        return data, labels


    def __load_test_data(self, data_dir):
        filename = os.path.join(data_dir, "test_batch")
        tmp = self._unpickle(filename)
        data = np.ndarray(shape=(0, 32 * 32 * 3), dtype=np.uint8)
        labels = np.ndarray(shape=0, dtype=np.int64)

        data = np.append(data, tmp[b'data'], axis=0)
        labels = np.append(labels, tmp[b'labels'])

        data = np.reshape(data, [-1, 32, 32, 3], 'F').transpose((0, 2, 1, 3))
        print('load test data: test_batch')
        return data, labels

    def write(self):
        train_data, train_labels = self.__load_train_data(self.data_dir)
        test_data, test_labels = self.__load_test_data(self.data_dir)
        if not os._exists(self.out_dir):
            os.makedirs(self.out_dir)
        def w(data, labels, name):
            writer = tf.python_io.TFRecordWriter(os.path.join(self.out_dir, name))
            for i in range(data.shape[0]):
                image = data[i, :].tobytes()
                label = labels[i]
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': self._bytes_feature(image),
                    'label': self._int64_feature(label),
                }))
                writer.write(example.SerializeToString())
            writer.close()
        w(train_data, train_labels, 'train.tfrecords')
        w(test_data, test_labels, 'test.tfrecords')

class CifarReader(TFRecordsReader):

    def __init__(self, batch_size):
        super(CifarReader, self).__init__(batch_size)
    def _parser(self, record):
        example = tf.parse_single_example(record,
                                          features={
                                              'image': tf.FixedLenFeature([], dtype=tf.string),
                                              'label': tf.FixedLenFeature([], dtype=tf.int64)},
                                          name='feature')
        image = tf.decode_raw(example['image'], tf.int8)
        image = tf.cast(tf.reshape(image, [32, 32, 3]), tf.float32)
        label = example['label']
        return image, label
    def _post_process(self, iterator):
        return iterator.get_next()


class styleTransfer(object):
    def __init__(self):
        with tf.contrib.layers.arg_scope([tf.contrib.layers.conv2d], {'trainable': False}):
            self.loss_net = alexnet.AlexNet(x, npy_path='./npy/alexnet.npy')
    def loss(self):
        endpoints = self.loss_net.endpoints
if __name__ == '__main__':
    # x = tf.placeholder(dtype=tf.float32, shape=[None, 227, 227, 3])

    # CifarConverter('../cifar-10-batches-py', '../tfrecords/cifar-10').write()
    images, labels = CifarReader(1).read('../tfrecords/cifar-10/train.tfrecords', 1, 1)
    c = 1

