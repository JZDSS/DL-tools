from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import time

import tensorflow as tf
import tensorflow.contrib.losses as losses
import numpy as np
from nets.resnet import *
import cfg

flags = cfg.FLAGS

def main(_):
    net = ResNet(flags)
    x = tf.placeholder(dtype=tf.float32, shape=(128, 64, 64, 3))
    net.build(x)
    print(net)


if __name__ == '__main__':
    tf.app.run()

