from resnet import *

import tensorflow as tf

tf_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
r=ResNet(num_classes = 10 , input_placeholder=tf_input)

for i in r.model_var_list[:]:
    print i.name
