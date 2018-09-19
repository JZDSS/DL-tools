import tensorflow as tf
import numpy as np
import basenets
import cv2
import time


def main(_):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 227, 227, 3))

    net = basenets.AlexNet({'images': x}, npy_path='../npy/alexnet.npy')
    init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)

    # test multiple images
    img1 = cv2.imread('../images/dog.jpg')
    img1 = cv2.resize(img1, (227, 227))

    img2 = cv2.imread('../images/lion.jpg')
    img2 = cv2.resize(img2, (227, 227))
    img = np.stack([img1, img2], 0)

    prediction = tf.argmax(net.outputs['logits'], axis=1)

    # use the following codes for testing one image
    # img1 = cv2.imread('../images/dog.jpg')
    # img1 = cv2.resize(img1, (227, 227))
    #
    # img = np.expand_dims(img1, 0)
    #
    # prediction = tf.argmax(net.outputs['logits'], axis=0)

    with tf.Session() as sess:
        sess.run(init_ops)
        while True:
            a = time.time()
            sess.run(net.endpoints['fc8'], feed_dict={x: img})
            b = time.time()
            print('1', b-a)
            b = time.time()
            sess.run(net.endpoints, feed_dict={x: img})
            c = time.time()
            print('2', c - b)
        print('Category numbers are:', sess.run(prediction, feed_dict={x: img}))
        print('Please look up `test/imagenet_classes.txt` for exact category names!')


if __name__ == "__main__":
    tf.app.run()