from resnet import *

import tensorflow as tf

from data_utils import get_CIFAR10_data
data = get_CIFAR10_data()
for k, v in data.iteritems():
    if 'X' in k:
        data[k] = np.einsum('ijkl->iklj', v)

    print '%s: ' % k, v.shape



tf_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
r=ResNet(num_classes = 10 , input_placeholder=tf_input)

for i in r.conv_var_list[:30]:
    print i.name
