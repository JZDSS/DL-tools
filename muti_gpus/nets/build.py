import tensorflow as tf
def net(x, FLAGS, is_training, num_classes):
    type = FLAGS.type
    if type == 'resnet':
        from muti_gpus.nets import resnet as my_net
    else:
        raise RuntimeError('Type error!!')

    y = my_net.build_net(x, FLAGS.blocks, is_training, FLAGS)
    return y