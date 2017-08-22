import numpy as np
import tensorflow as tf

from resnet import ResNet
import os

# Get CIFAR10 DATA
from data_utils import get_CIFAR10_data
data = get_CIFAR10_data()
for k, v in data.iteritems():
    if 'X' in k:
        data[k] = np.einsum('ijkl->iklj', v)/128.

    print '%s: ' % k, v.shape


with tf.Session() as sess:
    image_size = 32
    images = tf.placeholder(tf.float32, [None, 32, 32, 3])
    true_out = tf.placeholder(tf.float32, [None, 10])
    r = ResNet(input_placeholder=images, num_classes=10, is_training=False)
    print('ResNet graph build, with # variables: %d' % r.get_var_count())
    model_var = tf.global_variables()
    print('num var in model : %d' % len(model_var))

    # Define Cost, Trainer
    with tf.variable_scope('Loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(logits=r.logits,
                                                       labels=true_out)
        cost = tf.reduce_mean(xent, name='xent')       
        #cost += r.weight_decay()
        tf.summary.scalar('cross_entropy', cost)
        correct_prediction = tf.equal(tf.argmax(r.logits,1),
                                      tf.argmax(true_out,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', accuracy)


    total_var = tf.global_variables()
    print('total num var in model : %d' % len(total_var))

    saver = tf.train.Saver(model_var)
    ckpt = tf.train.get_checkpoint_state('Model/old')
    #ckpt = tf.train.get_checkpoint_state(os.path.dirname('/ResNet/Model'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)

    uninit_var = sess.run(tf.report_uninitialized_variables(tf.global_variables()))
#    import pdb; pdb.set_trace()
    print('un initialized var in model : %d' % len(uninit_var))
#    sess.run(tf.initialize_variables(uninit_var))
    x_batch = np.array(data['X_val'])
    y_batch = tf.one_hot(data['y_val'], 10).eval(session=sess)
    print('accuracy : ',sess.run(accuracy,feed_dict={images: x_batch, true_out:
                                                     y_batch}))

