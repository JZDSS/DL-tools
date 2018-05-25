import tensorflow as tf
import numpy as np
from ssd.protos import model_pb2
from google.protobuf import text_format
import model_builder


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
    anchor_config['src'] = src
    return anchor_config


def get_image_config(image):
    image_config = {}
    image_config['height'] = image.height
    image_config['width'] = image.width
    image_config['min_jaccard_overlap'] = image.minimum_jaccard_overlap
    image_config['aspect_ratio_range'] = (image.min_aspect_ratio, image.max_aspect_ratio)
    image_config['area_range']  = (image.min_area, image.max_area)
    image_config['flip'] = image.flip
    return image_config


model = model_pb2.Model()

with tf.gfile.GFile('/home/yqi/Desktop/workspace/PycharmProjects/DL-tools/ssd/ssdd.config', "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, model)

anchor_config = get_anchor_config(model.anchor_list)


image_config = get_image_config(model.image)
model_config = {}
model_config['name'] = model.name
model_config['type'] = model.type

with tf.get_default_graph():
    builder = model_builder.ModelBuilder('SSD_ALEX', net_name='SSD_alexnet', image_config=image_config, fake=True)
    builder(num_classes=3, anchor_config=anchor_config)

feature_map_size = [builder.model.endpoints[k].get_shape().as_list()[1:3] for k in anchor_config['src']]
anchor_config['feature_map_size'] = feature_map_size

tf.reset_default_graph()
