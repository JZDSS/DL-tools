import os
import tensorflow as tf
import trainer

class DefaultTrainer(trainer.Trainer):
    def __init__(self, net, gpus=[0]):
        super(DefaultTrainer, self).__init__(net, gpus)

    def _learning_rate(self, cfg):
        pass
    def _optimizer(self, cfg):
        pass

import cfg
from nets.resnet import *
flags = cfg.FLAGS
def main():
    net = ResNet(flags)
    mtrainer = DefaultTrainer(net)
    mtrainer.build_graph(None, None)


if __name__ == '__main__':
    main()