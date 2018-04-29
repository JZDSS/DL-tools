import tensorflow as tf
import numpy as np
import nets

def main(_):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 227, 227, 3))

    net = nets.AlexNet(x, npy_path='../npy/alexnet.npy')
    init_ops = tf.get_collection(tf.GraphKeys.INIT_OP)

    import cv2
    img1 = cv2.imread('../images/dog.jpg')
    img1 = cv2.resize(img1, (227, 227))

    img2 = cv2.imread('../images/lion.jpg')
    img2 = cv2.resize(img2, (227, 227))
    img = np.stack([img1, img2], 0)

    prediction = tf.argmax(net.logits, 1)

    with tf.Session() as sess:
        sess.run(init_ops)
        print(sess.run(prediction, feed_dict={x: img}))


if __name__ == "__main__":
    tf.app.run()