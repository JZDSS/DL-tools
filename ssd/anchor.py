import tensorflow as tf
import numpy as np


class Anchor(object):

    def __init__(self):
        super(Anchor, self).__init__()
        self.min_scale = 0.2
        self.max_scale = 0.9
        self.aspect_ratios = [[1 / 2, 2],
                              [1 / 3, 1 / 2, 2, 3],
                              [1 / 3, 1 / 2, 2, 3],
                              [1 / 3, 1 / 2, 2, 3],
                              [1 / 2, 2],
                              [1 / 2, 2]]
        self.n = 6
        a = np.linspace(self.min_scale, self.max_scale, self.n)
        a = 1


anchor_scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]  # of original image size
# s_k^{'} = sqrt(s_k*s_k+1)
ext_anchor_scales = [0.26, 0.28, 0.43, 0.60, 0.78, 0.98]  # of original image size
# omitted aspect ratio 1
aspect_ratios = [[1 / 2, 2],  # conv4_3
                         [1 / 3, 1 / 2, 2, 3],  # conv7
                         [1 / 3, 1 / 2, 2, 3],  # conv8_2
                         [1 / 3, 1 / 2, 2, 3],  # conv9_2
                         [1 / 2, 2],  # conv10_2
                         [1 / 2, 2]]  # conv11_2
feature_map_size = [[36, 36],
                    [17, 17],
                    [9, 9],
                    [5, 5],
                    [3, 3],
                    [1, 1]]