import tensorflow as tf
import numpy as np
from ssd.protos import anchor_list_pb2
from google.protobuf import text_format

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


anchor_list = anchor_list_pb2.AnchorList()
# print(anchor.min_scale)
# print(anchor.max_scale)
with tf.gfile.GFile('ssd.config', "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, anchor_list)

src = []
aspect_ratios = []

for anchor in anchor_list.anchor:
    ratio =[r for r in anchor.aspect_ratio]
    src.append(anchor.src)
    aspect_ratios.append(ratio)

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