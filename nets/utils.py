import tensorflow as tf


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def read_from_tfrecord(tfrecord_file_queue, flags):
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'label': tf.FixedLenFeature([], tf.string),
                                                    'patch_raw': tf.FixedLenFeature([], tf.string)
                                                }, name='features')
    image = tf.decode_raw(tfrecord_features['patch_raw'], tf.uint8)
    ground_truth = tf.decode_raw(tfrecord_features['label'], tf.int32)

    image = tf.cast(tf.reshape(image, [flags.patch_size, flags.patch_size, 3]), tf.float32)
    if flags.aug:
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    ground_truth = tf.reshape(ground_truth, [1])
    return image, ground_truth


def input_pipeline(filenames, batch_size, flags, read_threads=2, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [read_from_tfrecord(filename_queue, flags)
                    for _ in range(read_threads)]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch