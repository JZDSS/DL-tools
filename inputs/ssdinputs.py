import tensorflow as tf
import numpy as np
from inputs import tfrecordinputs

# np.set_printoptions(threshold=np.inf)


class SSDInputs(tfrecordinputs.TFRecordsInputs):

    def __init__(self, config=None, fake=False, **kwargs):
        super(SSDInputs, self).__init__(config['train']['batch_size'], fake, **kwargs)
        image_config = config['image']
        anchor_config = config['anchor']
        self.anchor_scales = anchor_config['scales']
        self.aspect_ratios = anchor_config['aspect_ratios']
        self.ext_anchor_scales = anchor_config['extra_scales']
        self.feature_map_size = anchor_config['feature_map_size']
        self.size = (image_config['height'], image_config['width'])
        self.channels = image_config['channels']
        self.extra_anchor = anchor_config['extra_anchor']

    def _random_horizontally_flip_with_bbox(self, image, bboxes, seed=None):
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

    def _transform_bboxes(self, bboxes, bbox_ref):
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

    def _random_crop_with_bbox(self, img, bboxes, labels, minimum_jaccard_overlap=0.7,
                              aspect_ratio_range=(0.5, 2), area_range=(0.1, 1.0),
                              seed=None, seed2=None):
        """
        Random crop the image and transfrom the bounding boxes to associated with the cropped image.

        :param img: The original image, a 3-D Tensor with shape [height, width, channels].
        :param bboxes: 2-D Tensor of type tf.float32 with shape [num_boxes, 4], the coordinates  of 2ed dimension
            is ordered [min_y, min_x, max_y, max_x], which are normalized to [0, 1] by dividing image height and width.
        :param labels: Labels.
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
            bboxes, labels, num_boxes = self._drop_small_bboxes(bboxes, bbox_for_slice, labels)
            transformed_bboxes = self._transform_bboxes(bboxes, bbox_for_slice)
        return cropped_image, transformed_bboxes, labels, num_boxes

    def _resize(self, image, size):
        with tf.name_scope('resize'):
            image = tf.image.resize_images(tf.expand_dims(image, 0), size)
            image = tf.reshape(image, tf.stack([size[0], size[1], 3]))
        return image

    def _drop_small_bboxes(self, bboxes, bbox_for_slice, labels):
        bbox_for_slice = tf.reshape(bbox_for_slice, [-1])
        def intersection(bbox1, bbox2):
            # int for intersection
            int_ymin = tf.maximum(bbox1[:, 0], bbox2[0])
            int_xmin = tf.maximum(bbox1[:, 1], bbox2[1])
            int_ymax = tf.minimum(bbox1[:, 2], bbox2[2])
            int_xmax = tf.minimum(bbox1[:, 3], bbox2[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)

            # vol for volume.
            vol_int = h * w
            return vol_int
        vol_int = intersection(bboxes, bbox_for_slice)
        vol_bboxes = (bboxes[:, 3] - bboxes[:, 1] )*(bboxes[:, 2] - bboxes[:, 0])
        scores = vol_int / vol_bboxes
        mask = scores > 0.5
        bboxes = tf.boolean_mask(bboxes, mask)
        labels = tf.boolean_mask(labels, mask)
        num_boxes = tf.reduce_sum(tf.cast(mask, tf.int32))
        return bboxes, labels, num_boxes

    def _pre_process(self, image, bboxes, size, labels):
        with tf.name_scope('pre_process'):
            image, bboxes, labels, num_boxes = self._random_crop_with_bbox(image, bboxes, labels)
            image, bboxes = self._random_horizontally_flip_with_bbox(image, bboxes)
            image = self._resize(image, size)
        return image, bboxes, labels, num_boxes

    def decode(self, net_outputs):

        locations = net_outputs['location']
        logits = net_outputs['classification']

        probs = [tf.nn.softmax(c, axis=-1) for c in logits]
        labels = [tf.argmax(p, axis=-1) for p in probs]
        probs = [tf.reduce_max(p, axis=-1) for p in probs]
        masks = [tf.greater(l, 0) for l in labels]

        x, y, w, h = list(zip(*[tf.unstack(loc, axis=-1) for loc in locations]))

        # x = [tf.multiply(t, mm) for t, mm in zip(x, masks)]
        # y = [tf.multiply(t, mm) for t, mm in zip(y, masks)]
        # w = [tf.multiply(t, mm) for t, mm in zip(w, masks)]
        # h = [tf.multiply(t, mm) for t, mm in zip(h, masks)]

        # for every feature map
        box_list = []
        prob_list = []
        for i, size in enumerate(self.feature_map_size):
            # d for default boxes(anchors)
            # x and y coordinate of centers of every cell, normalized to [0, 1]
            d_cy, d_cx = np.mgrid[0:size[0], 0:size[1]].astype(np.float32)
            d_cx = (d_cx + 0.5) / size[1]
            d_cy = (d_cy + 0.5) / size[0]
            d_cx = np.expand_dims(d_cx, axis=-1)
            d_cy = np.expand_dims(d_cy, axis=-1)

            # calculate width and heights
            d_w = []
            d_h = []
            scale = self.anchor_scales[i]
            # two aspect ratio 1 anchor scales
            if self.extra_anchor[i]:
                d_w.append(self.ext_anchor_scales[i])
                d_h.append(self.ext_anchor_scales[i])
            # other anchor scales
            for ratio in self.aspect_ratios[i]:
                d_w.append(scale * np.sqrt(ratio))
                d_h.append(scale / np.sqrt(ratio))
            d_w = np.array(d_w, dtype=np.float32)
            d_h = np.array(d_h, dtype=np.float32)

            # d_ymin = d_cy - d_h / 2
            # d_ymax = d_cy + d_h / 2
            # d_xmin = d_cx - d_w / 2
            # d_xmax = d_cx + d_w / 2
            # vol_anchors = (d_xmax - d_xmin) * (d_ymax - d_ymin)

            g_cx = x[i]
            g_cy = y[i]
            g_w = w[i]
            g_h = h[i]
            mask = tf.contrib.layers.flatten(masks[i])
            g_cx = tf.boolean_mask(tf.contrib.layers.flatten(g_cx * d_w + d_cx), mask)
            g_cy = tf.boolean_mask(tf.contrib.layers.flatten(g_cy * d_h + d_cy), mask)
            g_w = tf.boolean_mask(tf.contrib.layers.flatten(tf.exp(g_w) * d_w), mask)
            g_h = tf.boolean_mask(tf.contrib.layers.flatten(tf.exp(g_h) * d_h), mask)

            x_min = g_cx - g_w / 2
            x_max = g_cx + g_w / 2
            y_min = g_cy - g_h / 2
            y_max = g_cy + g_h / 2

            box = tf.stack([y_min, x_min, y_max, x_max], axis=-1)
            prob = tf.boolean_mask(tf.contrib.layers.flatten(probs[i]), mask)
            box_list.append(box)
            prob_list.append(prob)

        box = tf.concat(box_list, axis=0)
        prob = tf.concat(prob_list, axis=0)
        return box, prob




    def _bounding_boxes2ground_truth(self, bboxes, labels, anchor_scales, ext_anchor_scales,
                                    aspect_ratios, feature_map_size, num_boxes, threshold=0.5):
        """

        :param bboxes: [num_boxes, 4]
        :param anchor_scales: [num_anchors]
        :param ext_anchor_scales: [num_anchors]
        :param aspect_ratios: [num_anchors, ?]
        :param feature_map_size:
        :return:
        """
        locations_all = []
        labels_all = []

        # for every feature map
        for i, size in enumerate(feature_map_size):
            # d for default boxes(anchors)
            # x and y coordinate of centers of every cell, normalized to [0, 1]
            d_cy, d_cx = np.mgrid[0:size[0], 0:size[1]].astype(np.float32)
            d_cx = (d_cx + 0.5) / size[1]
            d_cy = (d_cy + 0.5) / size[0]
            d_cx = np.expand_dims(d_cx, axis=-1)
            d_cy = np.expand_dims(d_cy, axis=-1)

            # calculate width and heights
            d_w = []
            d_h = []
            scale = anchor_scales[i]
            # two aspect ratio 1 anchor scales
            if self.extra_anchor[i]:
                d_w.append(ext_anchor_scales[i])
                d_h.append(ext_anchor_scales[i])
            # other anchor scales
            for ratio in aspect_ratios[i]:
                d_w.append(scale * np.sqrt(ratio))
                d_h.append(scale / np.sqrt(ratio))
            d_w = np.array(d_w, dtype=np.float32)
            d_h = np.array(d_h, dtype=np.float32)

            d_ymin = d_cy - d_h/2
            d_ymax = d_cy + d_h/2
            d_xmin = d_cx - d_w/2
            d_xmax = d_cx + d_w/2
            vol_anchors = (d_xmax - d_xmin) * (d_ymax - d_ymin)

            def calc_jaccard(bbox):
                """
                Calculate jaccard overlap with all feature_map_size[0]*feature_map_size[1]*num_anchors anchors
                :param bbox: [d_ymin, d_xmin, d_ymax, d_xmax]
                :return: jaccard overlap matrix with shape [feature_map_size[0], feature_map_size[1], num_anchors]
                """
                # int for intersection
                int_ymin = tf.maximum(d_ymin, bbox[0])
                int_xmin = tf.maximum(d_xmin, bbox[1])
                int_ymax = tf.minimum(d_ymax, bbox[2])
                int_xmax = tf.minimum(d_xmax, bbox[3])
                h = tf.maximum(int_ymax - int_ymin, 0.)
                w = tf.maximum(int_xmax - int_xmin, 0.)

                # vol for volume.
                vol_int = h * w
                vol_union = vol_anchors - vol_int \
                            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                jaccard = tf.div(vol_int, vol_union)

                return jaccard

            shape = size + [len(d_w)]
            final_labels = tf.zeros(shape, dtype=tf.int64)
            final_scores = tf.zeros(shape, dtype=tf.float32)
            final_ymin = tf.zeros(shape, dtype=tf.float32)
            final_xmin = tf.zeros(shape, dtype=tf.float32)
            final_ymax = tf.zeros(shape, dtype=tf.float32)
            final_xmax = tf.zeros(shape, dtype=tf.float32)
            n_box = 0

            def cond(n_box, final_labels, final_scores, final_ymin, final_xmin, final_ymax, final_xmax):
                return n_box < num_boxes

            def body(n_box, final_labels, final_scores, final_ymin, final_xmin, final_ymax, final_xmax):
                bbox = bboxes[n_box, :]
                label = labels[n_box]

                jaccard = calc_jaccard(bbox)

                mask = tf.greater(jaccard, threshold)
                mask = tf.logical_and(mask, jaccard > final_scores)
                int_mask = tf.cast(mask, tf.int64)
                float_mask = tf.cast(mask, tf.float32)
                final_labels = int_mask * label + (1 - int_mask) * final_labels
                final_scores = tf.where(mask, jaccard, final_scores)

                final_ymin = float_mask * bbox[0] + (1 - float_mask) * final_ymin
                final_xmin = float_mask * bbox[1] + (1 - float_mask) * final_xmin
                final_ymax = float_mask * bbox[2] + (1 - float_mask) * final_ymax
                final_xmax = float_mask * bbox[3] + (1 - float_mask) * final_xmax
                n_box = n_box + 1
                return n_box, final_labels, final_scores, final_ymin, final_xmin, final_ymax, final_xmax

            n_box, final_labels, final_scores, final_ymin, final_xmin, final_ymax, final_xmax = \
                tf.while_loop(cond, body,
                              [n_box, final_labels, final_scores, final_ymin, final_xmin, final_ymax, final_xmax])

            g_cx = (final_xmax + final_xmin) / 2
            g_cy = (final_ymax + final_ymin) / 2
            g_w = final_xmax - final_xmin
            g_h = final_ymax - final_ymin

            g_cx = (g_cx - d_cx) / d_w
            g_cy = (g_cy - d_cy) / d_h
            mask = tf.not_equal(final_labels, 0)
            g_w = tf.where(mask, tf.log(g_w / d_w), g_w)
            g_h = tf.where(mask, tf.log(g_h / d_h), g_h)

            locations_all.append(tf.stack([g_cx, g_cy, g_w, g_h], -1))
            labels_all.append(final_labels)

        return locations_all, labels_all

    def _parser(self, record):
        tfrecord_features = tf.parse_single_example(record,
                                                    features={
                                                        'image_string': tf.FixedLenFeature([], dtype=tf.string),
                                                        'labels': tf.VarLenFeature(dtype=tf.int64),
                                                        'height': tf.FixedLenFeature([1], dtype=tf.int64),
                                                        'width': tf.FixedLenFeature([1], dtype=tf.int64),
                                                        'ymin': tf.VarLenFeature(dtype=tf.float32),
                                                        'xmin': tf.VarLenFeature(dtype=tf.float32),
                                                        'ymax': tf.VarLenFeature(dtype=tf.float32),
                                                        'xmax': tf.VarLenFeature(dtype=tf.float32)
                                                    }, name='features')
        # height = tfrecord_features['height']
        # width = tfrecord_features['width']
        image = tf.image.decode_jpeg(tfrecord_features['image_string'], channels=3)
        image = tf.reverse(image, axis=[-1])
        # image = tf.transpose(image, [2, 1, 0])
        # image.set_shape([height, width, 3])
        labels = tf.sparse_tensor_to_dense(tfrecord_features['labels'])
        ymin = tf.sparse_tensor_to_dense(tfrecord_features['ymin'])
        xmin = tf.sparse_tensor_to_dense(tfrecord_features['xmin'])
        ymax = tf.sparse_tensor_to_dense(tfrecord_features['ymax'])
        xmax = tf.sparse_tensor_to_dense(tfrecord_features['xmax'])
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        image, bboxes, labels, num_boxes = self._pre_process(image, bboxes, self.size, labels)
        #
        locations, labels = self._bounding_boxes2ground_truth(bboxes, labels, self.anchor_scales,
                                    self.ext_anchor_scales, self.aspect_ratios, self.feature_map_size,
                                    num_boxes, threshold=0.5)
        mean = tf.constant([123, 117, 104], dtype=image.dtype)
        mean = tf.reshape(mean, [1, 1, 3])
        image = image - mean
        image = image / 128.0


        return [image] + locations + labels

    def _post_process(self, iterator):
        if iterator is None and self.fake:
            net_inputs = {'images': tf.zeros(dtype=tf.float32, shape=(1, self.size[0], self.size[1], self.channels))}
            ground_truth = None
            return net_inputs, ground_truth
        e = iterator.get_next()
        [tensor.set_shape([self.batch_size] + tensor.get_shape().as_list()[1:]) for tensor in e]
        images = e[0]
        locations = list(e[1:len(e) // 2 + 1])
        labels = list(e[len(e) // 2 + 1:])
        net_inputs = {'images': images}
        ground_truth = {'locations': locations, 'labels': labels}
        return net_inputs, ground_truth

