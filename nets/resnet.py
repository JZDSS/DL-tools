import tensorflow as tf

from nets import net
from nets import layers


class ResNet(net.Net):
    def __init__(self, flags):
        super(ResNet, self).__init__('ResNet')
        self.subtype = flags.subtype
        self.blocks = flags.blocks
        self.num_gpus = len(set(flags.gpu.split(',')))
        self.blocks = flags.blocks
        self.is_training = tf.placeholder(tf.bool)
        self.architecture = []
        self.weight_decay = flags.weight_decay
        self.num_classes = flags.num_classes


    def _block(self, inputs, num_outputs, weight_decay, scope, is_training, down_sample=False):
        with tf.variable_scope(scope):
            num_inputs = inputs.get_shape().as_list()[3]

            res = layers.conv(inputs, num_outputs=num_outputs, kernel_size=[3, 3],
                              strides=[1, 2, 2, 1] if down_sample else [1, 1, 1, 1],
                              scope='conv1', b_norm=True, is_training=is_training, weight_decay=weight_decay)

            res = layers.conv(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                              scope='conv2', b_norm=True, is_training=is_training, weight_decay=weight_decay)
            if num_inputs != num_outputs:
                inputs = layers.conv(inputs, num_outputs=num_outputs, kernel_size=[1, 1], activation_fn=None,
                                     scope='short_cut', strides=[1, 2, 2, 1], b_norm=True, is_training=is_training,
                                     weight_decay=weight_decay)
            res = tf.nn.relu(res + inputs)
        return res

    def build(self, inputs):
        if self.subtype == 'vgg':
            pass
        elif self.subtype == 'cifar10':
            n = self.blocks
            # shape = x.get_shape().as_list()
            with tf.variable_scope('pre'):
                pre = layers.conv(inputs, num_outputs=16, kernel_size=[3, 3], scope='conv', b_norm=True,
                                  is_training=self.is_training,
                                  weight_decay=self.weight_decay)
                # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
            self.architecture.append('conv3/1\n')
            h = pre
            for i in range(1, n + 1):
                h = self._block(h, 16, self.weight_decay, '16_block{}'.format(i), self.is_training)
                self.architecture.append('16_block{}/1\n'.format(i))

            h = self._block(h, 32, self.weight_decay, '32_block1', self.is_training, True)
            self.architecture.append('32_block1/2\n')
            for i in range(2, n + 1):
                h = self._block(h, 32, self.weight_decay, '32_block{}'.format(i), self.is_training)
                self.architecture.append('32_block{}/1\n'.format(i))

            h = self._block(h, 64, self.weight_decay, '64_block1', self.is_training, True)
            self.architecture.append('64_block1/2\n')
            for i in range(2, n + 1):
                h = self._block(h, 64, self.weight_decay, '64_block{}'.format(i), self.is_training)
                self.architecture.append('64_block{}/1\n'.format(i))

            shape = h.get_shape().as_list()
            h = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool')
            self.architecture.append('avg_pool\n')

            h = layers.conv(h, num_outputs=self.num_classes, kernel_size=[1, 1], scope='fc1', padding='VALID',
                            b_norm=True, is_training=self.is_training, weight_decay=self.weight_decay, activation_fn=None)
            self.architecture.append('fc\n')

            return tf.reshape(h, [-1, self.num_classes])

    def build_batch_norm_update_ops(self):
        pass

    def __str__(self):
        return ''.join(self.architecture)