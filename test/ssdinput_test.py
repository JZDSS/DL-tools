import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from inputs.ssdinputs import SSDInputs
import tensorflow as tf
import numpy as np
from ssd.configure import Configure


def main():
    # tf.enable_eager_execution()
    config = Configure('/home/yqi/workspace/DL-tools/ssd/ssd_alex.config').get_config()
    ipt = SSDInputs(config, config['eval']['batch_size'])
    images, ground_truth = ipt.input_pipeline(os.path.join('../ssd', '*train.tfrecords'), 1, 1)
    with tf.Session() as sess:
        while True:
            print(images['images'])
            a = (images['images'].eval()[0, :] * 128 + 128).astype(np.uint8)
            import cv2
            # a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
            cv2.imshow('', a)
            cv2.waitKey(0)
if __name__ == '__main__':
    main()