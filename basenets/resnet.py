import tensorflow as tf
from basenets import net

layers = tf.layers

class ResNet(net.Net):
    """
    implement some ResNets mentioned by Deep Residual Learning for Image Recognition.

    Attributes:
        type: An optional string from: "vgg", "cifar10", "100" and "151". Defaults to None.
        The type of ResNet. With type 'vgg', the net is modified from vgg16, and 'cifar10'
        means the net architecture is same with the net used in cifar10 experiments in the
        paper. Alternatively, '100' and '151' means 100 layers ResNet and 151 layers ResNet
        respectively.
        blocks:
        is_training:
        weight_decay:
        num_classes:
    """

    def __init__(self, inputs, type, weight_decay, num_classes, name='ResNet'):
        super(ResNet, self).__init__(name)
        self.type = type
        self.is_training = tf.placeholder(tf.bool)
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.inputs = inputs
        # self.output = None
        # self.probability = None


    def _block(self, inputs, num_outputs, weight_decay, scope, is_training, down_sample=False):
        with tf.variable_scope(scope):
            num_inputs = inputs.get_shape().as_list()[3]

            res = layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[3, 3],
                              strides=[1, 2, 2, 1] if down_sample else [1, 1, 1, 1],
                              scope='conv1', b_norm=True, is_training=is_training, weight_decay=weight_decay)

            res = layers.conv2d(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                              scope='conv2', b_norm=True, is_training=is_training, weight_decay=weight_decay)
            if num_inputs != num_outputs:
                inputs = layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[1, 1], activation_fn=None,
                                     scope='short_cut', strides=[1, 2, 2, 1], b_norm=True, is_training=is_training,
                                     weight_decay=weight_decay)
            res = tf.nn.relu(res + inputs)
        self.architecture.append(scope + '\n')
        return res

    def _vgg(self):
        inputs = self.inputs
        is_training = self.is_training
        weight_decay = self.weight_decay
        architecture = self.architecture
        block = self._block()
        num_classes = self.num_classes
        with tf.variable_scope('pre'):
            pre = layers.conv2d(inputs, num_outputs=64, kernel_size=[7, 7], scope='conv', b_norm=True,
                              is_training=is_training,
                              weight_decay=weight_decay)
            architecture.append('conv7x7/1')
            pre = tf.contrib.layers.max_pool2d(pre, [3, 3], stride=2, padding='SAME', scope='pool')  # 32
            architecture.append('maxpool3x3/2')
        h = pre
        for i in range(1, 4):
            h = block(h, 64, weight_decay, '64_block{}'.format(i), is_training)

        h = block(h, 128, weight_decay, '128_block_s2', is_training, True)
        for i in range(1, 4):
            h = block(h, 128, weight_decay, '128_block{}'.format(i), is_training)

        h = block(h, 256, weight_decay, '256_block_s2', is_training, True)
        for i in range(1, 6):
            h = block(h, 256, weight_decay, '256_block{}'.format(i), is_training)

        h = block(h, 512, weight_decay, '512_block_s2', is_training, True)
        for i in range(1, 3):
            h = block(h, 512, weight_decay, '512_block{}'.format(i), is_training)
        shape = h.get_shape().as_list()
        h = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool')
        architecture.append('avg_pool')
        shape = h.get_shape().as_list()
        h = layers.conv2d(h, num_outputs=num_classes, kernel_size=[shape[1], shape[2]], scope='fc1',
                        padding='VALID',
                        b_norm=True, is_training=is_training, weight_decay=weight_decay, activation_fn=None)
        architecture.append('fc')
        res = tf.reshape(h, [-1, num_classes])


    def build(self, inputs):
        res = []
        if self.type == 'vgg':
            self._vgg()
        elif self.type == 'cifar10':
            n = self.blocks
            # shape = x.get_shape().as_list()
            with tf.variable_scope('pre'):
                pre = layers.conv2d(inputs, num_outputs=16, kernel_size=[3, 3], scope='conv', b_norm=True,
                                  is_training=self.is_training,
                                  weight_decay=self.weight_decay)
                # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
            self.architecture.append('conv3x3/1\n')
            h = pre
            for i in range(1, n + 1):
                h = self._block(h, 16, self.weight_decay, '16_block{}'.format(i), self.is_training)
                # self.architecture.append('16_block{}/1\n'.format(i))

            h = self._block(h, 32, self.weight_decay, '32_block1', self.is_training, True)
            # self.architecture.append('32_block1/2\n')
            for i in range(2, n + 1):
                h = self._block(h, 32, self.weight_decay, '32_block{}'.format(i), self.is_training)
                # self.architecture.append('32_block{}/1\n'.format(i))

            h = self._block(h, 64, self.weight_decay, '64_block1', self.is_training, True)
            # self.architecture.append('64_block1/2\n')
            for i in range(2, n + 1):
                h = self._block(h, 64, self.weight_decay, '64_block{}'.format(i), self.is_training)
                # self.architecture.append('64_block{}/1\n'.format(i))

            shape = h.get_shape().as_list()
            h = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool')
            self.architecture.append('avg_pool\n')

            h = layers.conv2d(h, num_outputs=self.num_classes, kernel_size=[1, 1], scope='fc1', padding='VALID',
                            b_norm=True, is_training=self.is_training, weight_decay=self.weight_decay, activation_fn=None)
            self.architecture.append('fc\n')
            res = tf.reshape(h, [-1, self.num_classes])
        elif self.type == '100':
            pass
        elif self.type == '151':
            pass
        else:
            raise RuntimeError('Unknown ResNet type!')
        # self.output = res
        # self.probability = tf.nn.softmax(self.output)
        return res

    # def get_norm_update_ops(self):
    #     pass

    def __str__(self):
        return ''.join(self.architecture)