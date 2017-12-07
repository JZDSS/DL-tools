import tensorflow as tf
def net(x, is_training, FLAGS):
    type = FLAGS.type
    if type == 'resnet':
        from muti_gpus.nets import resnet as my_net
    else:
        raise RuntimeError('Type error!!')

    y = my_net.build_net(x, is_training, FLAGS)

    return y