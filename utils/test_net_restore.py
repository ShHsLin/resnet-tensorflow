from resnet import *

import tensorflow as tf

sess = tf.Session()

tf_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
r=ResNet(num_classes = 10 , input_placeholder=tf_input, is_training=False)

for i in r.model_var_list[:]:
    print i.name

saver = tf.train.Saver(r.model_var_list)
ckpt = tf.train.get_checkpoint_state('Model/old')
#ckpt = tf.train.get_checkpoint_state(os.path.dirname('/ResNet/Model'))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess,ckpt.model_checkpoint_path)

uninit_var = sess.run(tf.report_uninitialized_variables(tf.global_variables()))
#    import pdb; pdb.set_trace()
print('un initialized var in model : %d' % len(uninit_var))

