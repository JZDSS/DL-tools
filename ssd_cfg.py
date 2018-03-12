import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_string('data_dir', '../SPCup/patches', 'data direction')
flags.DEFINE_string('log_dir', './logs', 'log direction')
flags.DEFINE_string('ckpt_dir', './ckpt', 'check point direction')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('max_steps', 192000, 'max steps')
flags.DEFINE_integer('start_step', 0, 'start steps')
flags.DEFINE_string('model_name', 'ssd_model', '')
flags.DEFINE_string('gpu', '3', '')
flags.DEFINE_string('out_file', 'logs/ssd_train_log', '')
flags.DEFINE_integer('patch_size', 64, '')


FLAGS = flags.FLAGS