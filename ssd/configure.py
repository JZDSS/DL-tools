import tensorflow as tf
import numpy as np
from ssd.protos import model_pb2
from google.protobuf import text_format
from builder import model_builder
from inputs.ssdinputs import SSDInputs


class Configure(object):

    def __init__(self, path='/home/yqi/Desktop/workspace/PycharmProjects/DL-tools/ssd/ssd_mobile.config'):
        self.config_file = path

    def _get_anchor_config(self, anchor_list):
        src = []
        aspect_ratios = []
        for anchor in anchor_list.anchor:
            ratio =[r for r in anchor.aspect_ratio]
            src.append(anchor.src)
            aspect_ratios.append(ratio)
        num_features = len(src)
        method = anchor_list.method
        if method == "linear":
            anchor_scales = np.linspace(anchor_list.min_scale, anchor_list.max_scale, num_features).tolist()
            anchor_scales.append(2*anchor_scales[-1] - anchor_scales[-2])
        elif method == "exp":
            q = (anchor_list.max_scale / anchor_list.min_scale) ** (1. / (len(aspect_ratios) - 1))
            anchor_scales = [anchor_list.min_scale]
            for i in range(len(aspect_ratios)):
                s = anchor_scales[-1] * q
                s = s if s <=1 else 0.99
                anchor_scales.append(s)
        else:
            print("unknown method!")
            raise RuntimeError
        ext_anchor_scales = []
        for i in range(num_features):
            ext_anchor_scales.append(np.sqrt(anchor_scales[i] * anchor_scales[i + 1]))
        anchor_scales.pop()
        anchor_config = {}
        anchor_config['scales'] = anchor_scales
        anchor_config['aspect_ratios'] = aspect_ratios
        anchor_config['extra_scales'] = ext_anchor_scales
        # boolean
        anchor_config['extra_anchor'] = [a.extra_anchor for a in anchor_list.anchor]
        anchor_config['src'] = src
        anchor_config['feature_map_size'] = None
        return anchor_config


    def _get_image_config(self, image):
        image_config = {}
        image_config['height'] = image.height
        image_config['width'] = image.width
        image_config['channels'] = image.channels
        image_config['min_jaccard_overlap'] = image.minimum_jaccard_overlap
        image_config['aspect_ratio_range'] = (image.min_aspect_ratio, image.max_aspect_ratio)
        image_config['area_range']  = (image.min_area, image.max_area)
        image_config['flip'] = image.flip
        image_config['path'] = image.path
        return image_config

    def _get_model_config(self, model):
        model_config = {}
        model_config['name'] = model.name
        model_config['type'] = model.type
        model_config['num_classes'] = model.num_classes
        if model.npy_path is not None:
            model_config['npy_path'] = model.npy_path
        else:
            model_config['npy_path'] = None
        return model_config

    def _get_train_config(self, train):
        train_config = {}
        train_config['batch_size'] = train.batch_size
        train_config['log_dir'] = train.log_dir
        train_config['ckpt_dir'] = train.ckpt_dir
        train_config['num_epochs'] = train.num_epochs
        train_config['image'] = self._get_image_config(train.image)
        train_config['weight_decay'] = train.weight_decay
        return train_config

    def _get_eval_config(self, eval):
        eval_config = {}
        eval_config['batch_size'] = eval.batch_size
        eval_config['log_dir'] = eval.log_dir
        eval_config['ckpt_dir'] = eval.ckpt_dir
        eval_config['num_epochs'] = eval.num_epochs if eval.num_epochs != 0 else None
        eval_config['image'] = self._get_image_config(eval.image)
        return eval_config

    def get_config(self):
        model = model_pb2.Model()

        with tf.gfile.GFile(self.config_file, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, model)

        anchor_config = self._get_anchor_config(model.anchor_list)

        model_config = self._get_model_config(model)

        train_config = self._get_train_config(model.train)

        eval_config = self._get_eval_config(model.eval)
        # with tf.get_default_graph():
        with tf.Graph().as_default():
            builder = model_builder.ModelBuilder({'model': model_config,
                                                  'train': train_config,
                                                  'anchor': anchor_config,
                                                  'eval': eval_config}, fake=True, input_class=SSDInputs)
            # builder()

            feature_map_size = [builder.model.endpoints[k].get_shape().as_list()[1:3] for k in anchor_config['src']]
            anchor_config['feature_map_size'] = feature_map_size

        # tf.reset_default_graph()
        config = {}
        config['model'] = model_config
        config['anchor'] = anchor_config
        config['train'] = train_config
        config['eval'] = eval_config
        return config

