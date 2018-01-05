import tensorflow as tf
def net(x, is_training, FLAGS):
    type = FLAGS.type
    if type == 'resnet':
        from multi_gpus.nets import resnet as my_net
    elif type == 'resnet2':
        from multi_gpus.nets import resnet2 as my_net
    elif type == 'resnet3':
        from multi_gpus.nets import resnet3 as my_net
    elif type == 'resnet_pool':
        from multi_gpus.nets import resnet_pool as my_net
    elif type == 'fuse':
        from multi_gpus.nets import fuse as my_net
    elif type == 'cluster':
        from multi_gpus.nets import cluster as my_net
    elif type == 'bcp':
        from multi_gpus.nets import bcp as my_net
    elif type == 'res34':
        from multi_gpus.nets import res34 as my_net
    else:
        raise RuntimeError('Type error!!')

    y = my_net.build_net(x, is_training, FLAGS)

    return y