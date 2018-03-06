import tensorflow as tf


def smooth_l1_loss(x):
    """
    Computes smooth l1 loss: x^2 / 2 if abs(x) < 1, abs(x) - 0.5 otherwise and add to tf.GraphKeys.LOSSES collection.
    See [Fast R-CNN](https://arxiv.org/abs/1504.08083)
    :param x: An input Tensor to calculate smooth L1 loss.
    """
    square_loss = 0.5 * x ** 2
    absolute_loss = tf.abs(x)
    loss = tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss - 0.5)
    tf.add_to_collection(tf.GraphKeys.LOSSES, loss)


def random_horizontally_flip_with_bbox(image, bboxes, seed=None):
    """

    :param image:
    :param bboxes:
    :param seed:
    :return:
    """
    def flip_bboxes(bboxes):
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                           bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes
    with tf.name_scope('random_flip'):
        uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = tf.less(uniform_random, .5)
        image = tf.cond(mirror_cond,
                           lambda: tf.reverse(image, [1]),
                           lambda: image)
        bboxes = tf.cond(mirror_cond,
                         lambda: flip_bboxes(bboxes),
                         lambda: bboxes)
    return image, bboxes


def _transform_bboxes(bboxes, bbox_ref):
    bbox_ref = tf.reshape(bbox_ref, [-1])
    v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
    transformed_bboxes = bboxes - v
    s = tf.stack([bbox_ref[2] - bbox_ref[0],
                  bbox_ref[3] - bbox_ref[1],
                  bbox_ref[2] - bbox_ref[0],
                  bbox_ref[3] - bbox_ref[1]])
    transformed_bboxes = transformed_bboxes / s
    transformed_bboxes = tf.nn.relu(transformed_bboxes)
    transformed_bboxes = 1 - tf.nn.relu(1 - transformed_bboxes)
    return transformed_bboxes


def random_crop_with_bbox(img, bboxes, minimum_jaccard_overlap=0.7,
                          aspect_ratio_range=(0.5, 2), area_range=(0.1, 1.0),
                          seed=None, seed2=None):
    """
    Random crop the image and transfrom the bounding boxes to associated with the cropped image.

    :param img: The original image, a 3-D Tensor with shape [height, width, channels].
    :param bboxes: 2-D Tensor of type tf.float32 with shape [num_boxes, 4], the coordinates  of 2ed dimension
        is ordered [min_y, min_x, max_y, max_x], which are normalized to [0, 1] by dividing image height and width.
    :param minimum_jaccard_overlap: A Tensor of type float32. Defaults to 0.7. The cropped area of the image must
        contain at least this fraction of any bounding box supplied. The value of this parameter should be non-negative.
        In the case of 0, the cropped area does not need to overlap any of the bounding boxes supplied.
    :param aspect_ratio_range: An optional list of floats. Defaults to [0.5, 2]. The cropped area of the image
        must have an aspect ratio = width / height within this range.
    :param area_range: An optional list of floats. Defaults to [0.1, 1]. The cropped area of the image must contain
        a fraction of the supplied image within in this range.
    :return: Cropped image and transformed bounding boxes.
    """
    with tf.name_scope('random_crop'):
        begin, size, bbox_for_slice = tf.image.sample_distorted_bounding_box(
                                    tf.shape(img),
                                    bounding_boxes=tf.expand_dims(bboxes, 0),
                                    min_object_covered=minimum_jaccard_overlap,
                                    aspect_ratio_range=aspect_ratio_range,
                                    area_range=area_range,
                                    use_image_if_no_bounding_boxes=True,
                                    seed=seed, seed2=seed2)
        cropped_image = tf.slice(img, begin, size)
        transformed_bboxes = _transform_bboxes(bboxes, bbox_for_slice)
    return cropped_image, transformed_bboxes


def resize(image, size):
    with tf.name_scope('resize'):
        image = tf.image.resize_images(tf.expand_dims(image, 0), size)
        image = tf.squeeze(image)
    return image


def pre_process(image, bboxes, size):
    with tf.name_scope('pre_process'):
        image, bboxes = random_crop_with_bbox(image, bboxes)
        image, bboxes = random_horizontally_flip_with_bbox(image, bboxes)
        image = resize(image, size)
    return image, bboxes


def main():
    import cv2
    import numpy as np
    img = cv2.imread('/home/yqi/Desktop/workspace/PycharmProjects/VOCdevkit/VOC2007/JPEGImages/000005.jpg')
    bbx = [[0.562, 0.526, 0.904, 0.648],
           [0.650, 0.01, 0.997, 0.134]]
    x = tf.placeholder(tf.float32, shape=[None, None, 3])
    b = tf.placeholder(tf.float32, shape=[2, 4])
    processed_image, processed_bbox = pre_process(x, b, (300, 300))
    drawed = tf.image.draw_bounding_boxes(tf.expand_dims(processed_image, 0), tf.expand_dims(processed_bbox, 0))
    with tf.Session() as sess:
        while True:
            t, to_show = sess.run([processed_bbox, drawed], feed_dict={x: img, b: bbx})
            print(t)
            cv2.imshow("a", to_show.astype(np.uint8)[0])
            cv2.waitKey(0)

if __name__ == '__main__':
    main()