import tensorflow as tf
from basenets import *
from ssd.nets import *

class ModelBuilder(object):

    def __init__(self, net_type, net_name, image_config, fake=False, *args, **kwargs):
        self.net_type = net_type
        self.net_name = net_name
        self.image_config = image_config
        if fake:
            self.image = tf.placeholder(tf.float32, shape=(1, image_config['height'], image_config['width'], 3))
            self.ground_truth = None
        else:
            raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if self.net_type == 'ALEX':
            self.model = AlexNet(args, **kwargs)
        elif self.net_type == 'SSD_ALEX':
            self.model = SSD_AlexNet(self.image, ground_truth=self.ground_truth, **kwargs)

