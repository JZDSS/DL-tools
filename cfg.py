import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('data_dir', '../SPCup/patches', 'data direction')
flags.DEFINE_string('log_dir', './logs', 'log direction')
flags.DEFINE_string('ckpt_dir', './ckpt', 'check point direction')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('max_steps', 150000, 'max steps')
flags.DEFINE_integer('start_step', 0, 'start steps')
flags.DEFINE_string('model_name', 'model', '')
flags.DEFINE_string('gpu', '3', '')
flags.DEFINE_integer('blocks', 3, '')
flags.DEFINE_string('out_file', '', '')
flags.DEFINE_integer('patch_size', 64, '')
flags.DEFINE_string('type', '', '')
flags.DEFINE_integer('num_classes', 10, '')
flags.DEFINE_integer('num_branches', 0, '')
flags.DEFINE_boolean('p_relu', False, '')
flags.DEFINE_boolean('aug', False, '')

FLAGS = flags.FLAGS