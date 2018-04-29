from __future__ import print_function
from __future__ import division
import tensorflow as tf
import os
from xml.etree import ElementTree

txt2label = {
    'none': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
    'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11,
    'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16,
    'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
}


class converter(object):

    def __init__(self, data_dir, out_dir, prefix, num_tfrecords=10):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')
        self.out_dir = out_dir
        self.num_tfrecords = num_tfrecords
        self.images = os.listdir(self.img_dir)
        self.annotations = os.listdir(self.ann_dir)
        assert len(self.images) == len(self.annotations)
        self.images_per_file = len(self.images) // num_tfrecords + 1
        self.prefix = prefix

    def tolist(self, value):
        if not isinstance(value, list):
            value = [value]
        return value

    def __bytes_feature(self, value):
        value = self.tolist(value)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def __int64_feature(self, value):
        value = self.tolist(value)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def __float_feature(self, value):
        value = self.tolist(value)
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def convert(self):

        writer = tf.python_io.TFRecordWriter('test.tfrecords')

        for image in self.images:
            annotation = image.split('.')[0] + '.xml'
            image_path = os.path.join(self.img_dir, image)
            annotation_path = os.path.join(self.ann_dir, annotation)

            image_string = open(image_path, 'rb').read()

            root = ElementTree.parse(annotation_path).getroot()
            labels = []
            ymins = []
            xmins = []
            ymaxs = []
            xmaxs = []
            size_ele = root.find('size')
            shape = [int(size_ele.find('height').text),
                     int(size_ele.find('width').text),
                     int(size_ele.find('depth').text)]
            for obj in root.findall('object'):

                label = txt2label[obj.find('name').text]
                box_ele = obj.find('bndbox')
                ymin = int(box_ele.find('ymin').text)/shape[0]
                xmin = int(box_ele.find('xmin').text)/shape[1]
                ymax = int(box_ele.find('ymax').text)/shape[0]
                xmax = int(box_ele.find('xmax').text)/shape[1]
                ymins.append(ymin)
                xmins.append(xmin)
                ymaxs.append(ymax)
                xmaxs.append(xmax)
                labels.append(label)
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_string': self.__bytes_feature(image_string),
                'labels': self.__int64_feature(labels),
                'ymin': self.__float_feature(ymins),
                'xmin': self.__float_feature(xmins),
                'ymax': self.__float_feature(ymaxs),
                'xmax': self.__float_feature(xmaxs),
                'height': self.__int64_feature(shape[0]),
                'width': self.__int64_feature(shape[1])
            }))

            writer.write(example.SerializeToString())
        writer.close()

def main():
    c = converter('/home/yqi/Desktop/workspace/PycharmProjects/VOCdevkit/VOC2007',
              './tfrecords', 'voc2007train', 15)
    c.convert()

if __name__ == '__main__':
    main()

