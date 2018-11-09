import tensorflow as tf
import numpy as np
import basenets
import cv2
import time
import matplotlib.pyplot as plt


def main(_):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 227, 227, 3))

    net = basenets.AlexNet({'images': x}, npy_path='../npy/alexnet.npy')
    init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)

    # test multiple images
    img1 = cv2.imread('../images/dog.jpg')
    img1 = cv2.resize(img1, (227, 227))

    # img2 = cv2.imread('../images/lion.jpg')
    # img2 = cv2.resize(img2, (227, 227))

    img2 = np.roll(img1, 50, 0)
    img2 = np.roll(img2, 50, 1)
    img = np.stack([img1, img2], 0)
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()
    with tf.Session() as sess:
        sess.run(init_ops)
        # while True:
        feat = sess.run(net.endpoints['conv5'], feed_dict={x: img})

        # print('Category numbers are:', sess.run(prediction, feed_dict={x: img}))
        # print('Please look up `test/imagenet_classes.txt` for exact category names!')
    # feat = np.sum(feat, -1)
    for i in range(4096):
        y1 = feat[0, :, :, i]
        y2 = feat[1, :, :, i]
        plt.subplot(1,2,1)
        plt.imshow(y1)
        plt.subplot(1,2,2)
        plt.imshow(y2)
        plt.show()
    # x = range(4096)
    # y1 = feat[0, 0, 0]
    # y2 = feat[1, 0, 0]
    # plt.subplot(1, 2, 1)
    # plt.plot(x, y1)
    # plt.subplot(1, 2, 2)
    # plt.plot(x, y2)
    # plt.show()
    # plt.pause(0)
if __name__ == "__main__":
    tf.app.run()