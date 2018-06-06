import tensorflow as tf

from basenets import *
from ssd.nets import *


class ModelBuilder(object):

    def __init__(self, config, fake=False, *args, **kwargs):
        self.config = config
        self.model_config = config['model']
        self.image_config = config['image']
        self.train_config = config['train']
        self.fake = fake

    def __call__(self, *args, **kwargs):
        if self.fake:
            self.images = tf.placeholder(tf.float32, shape=(1, self.image_config['height'], self.image_config['width'], 3))
            self.ground_truth = None
        else:
            pipeline = kwargs['input_class'](self.config)
            filenames = tf.train.match_filenames_once(self.image_config['path'])
            self.images, self.ground_truth = pipeline.input_pipeline(filenames, self.train_config['batch_size'], 1)
            del kwargs['input_class']

        type = self.model_config['type']
        if type == 'ALEX':
            self.model = AlexNet(args, **kwargs)
        elif type == 'SSD_ALEX':
            self.model = SSD_AlexNet(self.images,
                                     self.model_config['num_classes'],
                                     self.ground_truth,
                                     name=self.model_config['name'], **kwargs)

