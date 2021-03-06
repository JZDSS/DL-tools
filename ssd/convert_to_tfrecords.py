from __future__ import print_function
from __future__ import division
import tensorflow as tf
import os
from xml.etree import ElementTree

# txt2label = {
#     'none': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
#     'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11,
#     'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16,
#     'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
# }


txt2label = {'fish': 1}


class converter(object):

    def __init__(self, data_dir, out_path, imageset):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')
        self.imageset = os.path.join(data_dir, 'ImageSets/') + imageset
        self.out_path = out_path
        self.ids = [x.strip() for x in open(self.imageset).readlines()]


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

        writer = tf.python_io.TFRecordWriter(self.out_path)

        for id in self.ids:
            annotation = id + '.xml'
            image = id + '.jpg'
            image_path = os.path.join(self.img_dir, image)
            name = annotation.split('.')[0].encode()
            annotation_path = os.path.join(self.ann_dir, annotation)
            try:
                image_string = open(image_path, 'rb').read()

                root = ElementTree.parse(annotation_path).getroot()
            except Exception as e:
                print(e)
                continue
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
                'name': self.__bytes_feature(name),
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
    c = converter('/home/yqi/Desktop/WineDownloads/大作业3/to student/FishData_subset/', './toy-test.tfrecords', 'test.txt')
    c.convert()

if __name__ == '__main__':
    main()

