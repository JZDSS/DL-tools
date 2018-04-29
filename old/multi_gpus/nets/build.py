import tensorflow as tf
def net(x, is_training, FLAGS):
    type = FLAGS.type
    if type == 'resnet':
        from multi_gpus.nets import resnet as my_net
    elif type == 'resnet2':
        from multi_gpus.nets import resnet2 as my_net
    elif type == 'resnet2_1':
        from multi_gpus.nets import resnet2_1 as my_net
    elif type == 'resnet3':
        from multi_gpus.nets import resnet3 as my_net
    elif type == 'resnet_pool':
        from multi_gpus.nets import resnet_pool as my_net
    elif type == 'fuse':
        from multi_gpus.nets import fuse as my_net
    elif type == 'cluster':
        from multi_gpus.nets import cluster as my_net
    elif type == 'cluster2':
        from multi_gpus.nets import cluster2 as my_net
    elif type == 'cluster3':
        from multi_gpus.nets import cluster3 as my_net
    elif type == 'bcp':
        from multi_gpus.nets import bcp as my_net
    elif type == 'res34':
        from multi_gpus.nets import res34 as my_net
    elif type == 'pre':
        from multi_gpus.nets import pre as my_net
    elif type == 'resnet2_prelu':
        from multi_gpus.nets import resnet2_prelu as my_net
    elif type == 'resnet2_elu':
        from multi_gpus.nets import resnet2_elu as my_net
    elif type == 'resnet2_prelu_channel':
        from multi_gpus.nets import resnet2_prelu_channel as my_net
    elif type == 'multi_scale':
        from multi_gpus.nets import multi_scale as my_net
    elif type == 'multi_scale2':
        from multi_gpus.nets import multi_scale2 as my_net
    elif type == 'multi_scale_deeper':
        from multi_gpus.nets import multi_scale_deeper as my_net
    elif type == 'deeper':
        from multi_gpus.nets import deeper as my_net
    else:
        raise RuntimeError('Type error!!')

    y = my_net.build_net(x, is_training, FLAGS)

    return y