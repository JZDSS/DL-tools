import tensorflow as tf

from basenets import *
from ssd.nets import *
from inputs.inputs import Inputs


class ModelBuilder(object):

    def __init__(self, config, mode='train', fake=False, input_class=Inputs, *args, **kwargs):
        self.config = config
        self.mode = mode
        self.model_config = config['model']
        self.image_config = config['image']
        self.train_config = config['train']
        self.fake = fake
        self.pipeline = None
        self.model = None
        self.input_class=input_class
        self.__call__()

    def __call__(self, *args, **kwargs):

        # input_class = kwargs['input_class']
        self.pipeline = self.input_class(self.config,
                               batch_size=self.train_config['batch_size'] if self.mode=='train' else 1,
                               fake=self.fake)
        filenames = self.image_config['path']
        self.net_inputs, self.ground_truth = self.pipeline.input_pipeline(filenames, len(filenames), 1,
                                                                     self.train_config['num_epochs'])
        # del kwargs['input_class']

        type = self.model_config['type']
        if type == 'ALEX':
            self.model = AlexNet(args, **kwargs)
        elif type == 'SSD_ALEX':
            self.model = SSDAlexNet(self.net_inputs,
                                    self.model_config['num_classes'],
                                    self.ground_truth,
                                    anchor_config=self.config['anchor'],
                                    name=self.model_config['name'],
                                    npy_path=self.model_config['npy_path'] if not self.fake else None, **kwargs)

