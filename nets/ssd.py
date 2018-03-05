import tensorflow as tf


def smooth_l1_loss(x):
    """Computes smooth l1 loss: x^2 / 2 if abs(x) < 1, abs(x) - 0.5 otherwise.

      See [Fast R-CNN](https://arxiv.org/abs/1504.08083)

      Args:
        x:
    """
    square_loss = 0.5 * x ** 2
    absolute_loss = tf.abs(x)
    loss = tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss - 0.5)
    tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
