import tensorflow as tf
import os


def read_from_tfrecord(tfrecord_file_queue):
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'label': tf.FixedLenFeature([], tf.string),
                                                    'patch_raw': tf.FixedLenFeature([], tf.string)
                                                }, name='features')
    image = tf.decode_raw(tfrecord_features['patch_raw'], tf.uint8)
    ground_truth = tf.decode_raw(tfrecord_features['label'], tf.int32)

    image = tf.cast(tf.reshape(image, [64, 64, 3]), tf.float32)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    ground_truth = tf.reshape(ground_truth, [1])
    return image, ground_truth


def input_pipeline(filenames, batch_size, read_threads=2, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [read_from_tfrecord(filename_queue)
                    for _ in range(read_threads)]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

images, labels = input_pipeline(
            tf.train.match_filenames_once(os.path.join('/data/qiyao/mix8', 'train', '*.tfrecords')),12800)
mean, var = tf.nn.moments(images, [0, 1, 2])
with tf.Session(config = config) as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(100):
        m, v = sess.run([mean, var])
        print(m)
    coord.request_stop()
    coord.join(threads)
