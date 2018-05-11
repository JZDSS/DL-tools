import tensorflow as tf
import numpy as np
from ssd.protos import model_pb2
from google.protobuf import text_format

model = model_pb2.Model()
# print(anchor.min_scale)
# print(anchor.max_scale)
with tf.gfile.GFile('ssd.config', "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, model)

def get_anchor_config(anchor_list):
    src = []
    aspect_ratios = []
    for anchor in anchor_list.anchor:
        ratio =[r for r in anchor.aspect_ratio]
        src.append(anchor.src)
        aspect_ratios.append(ratio)
    num_features = len(src)
    anchor_scales = np.linspace(anchor_list.min_scale, anchor_list.max_scale, num_features).tolist()
    anchor_scales.append(2*anchor_scales[-1] - anchor_scales[-2])
    ext_anchor_scales = []
    for i in range(num_features):
        ext_anchor_scales.append(np.sqrt(anchor_scales[i] * anchor_scales[i + 1]))
    anchor_scales.pop()
    anchor_config = {}
    anchor_config['scales'] = anchor_scales
    anchor_config['aspect_ratios'] = aspect_ratios
    anchor_config['extra_scales'] = ext_anchor_scales
    anchor_config['extra_anchor'] = [a.extra_anchor for a in anchor_list.anchor]
    return anchor_config

anchor_config = get_anchor_config(model.anchor_list)

def get_image_config(image):
    image_config = {}
    image_config['height'] = image.height
    image_config['width'] = image.height
    image_config['min_jaccard_overlap'] = image.minimum_jaccard_overlap
    image_config['aspect_ratio_range'] = (image.min_aspect_ratio, image.max_aspect_ratio)
    image_config['area_range']  = (image.min_area, image.max_area)
    image_config['flip'] = image.flip
    return image_config

image_config = get_image_config(model.image)

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