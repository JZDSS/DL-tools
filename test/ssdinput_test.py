from inputs.ssdinputs import SSDInputs
import tensorflow as tf
import numpy as np

def main():
    from ssd.configure import Configure
    import os
    # tf.enable_eager_execution()
    config = Configure('/home/yqi/Desktop/workspace/PycharmProjects/DL-tools/ssd/ssdd.config').get_config()
    ipt = SSDInputs(config, False)
    images, ground_truth = ipt.input_pipeline(os.path.join('../ssd', '*.tfrecords'), 1, 1)
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