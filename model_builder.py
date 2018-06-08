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

        input_class = kwargs['input_class']
        pipeline = input_class(self.config, fake=self.fake)
        filenames = self.image_config['path']
        self.net_inputs, self.ground_truth = pipeline.input_pipeline(filenames, len(filenames), 1)
        del kwargs['input_class']

        type = self.model_config['type']
        if type == 'ALEX':
            self.model = AlexNet(args, **kwargs)
        elif type == 'SSD_ALEX':
            self.model = SSDAlexNet(self.net_inputs,
                                    self.model_config['num_classes'],
                                    self.ground_truth,
                                    anchor_config=self.config['anchor'],
                                    name=self.model_config['name'], **kwargs)

