import numpy as np
import tensorflow as tf
from resnet import ResNet
import os



def image_pad_random_crop(x_batch, pad_size=4):
    batchsize, x_dim, y_dim, _ = x_batch.shape
    pad_image = np.zeros([batchsize, x_dim + pad_size * 2,
                          y_dim + pad_size * 2, 3])
    pad_image[:,pad_size:pad_size + x_dim,
              pad_size:pad_size+y_dim,:] = x_batch
    crop_ind = np.random.randint(2*pad_size, size=(2,batchsize))
    image = np.zeros(x_batch.shape)
    for i in xrange(batchsize):
        image[i] = pad_image[i,crop_ind[0,i]:crop_ind[0,i]+x_dim,
                             crop_ind[1,i]:crop_ind[1,i]+y_dim,:]

    np.flip(image[batchsize/2:batchsize], axis=2)
    return image


# Get CIFAR10 DATA
from data_utils import get_CIFAR10_data
data = get_CIFAR10_data()
for k, v in data.iteritems():
    if 'X' in k:
        data[k] = np.einsum('ijkl->iklj', v)

    print '%s: ' % k, v.shape


config = tf.ConfigProto(allow_soft_placement=True) # , log_device_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.90
config.gpu_options.allow_growth = True

with tf.device('/gpu:0'):
    with tf.Session(config=config) as sess:
        image_size = 32
        num_classes = 10
        images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        true_out = tf.placeholder(tf.float32, [None, num_classes])
        r = ResNet(input_placeholder=images, num_classes=num_classes, is_training=True)
        print('ResNet graph build, with # variables: %d' % r.get_var_count())

        # Define Cost, Trainer
        # This should move into the resnet.py
        with tf.variable_scope('CIFAR10'):
            xent = tf.nn.softmax_cross_entropy_with_logits(logits=r.logits,
                                                           labels=true_out)
            cost = tf.reduce_mean(xent, name='xent')
            cost += r.weight_decay(0.0001)
            tf.summary.scalar('cross_entropy', cost)
            correct_prediction = tf.equal(tf.argmax(r.logits,1),
                                          tf.argmax(true_out,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.summary.scalar('accuracy', accuracy)

         
        #train_step = tf.train.AdamOptimizer(0.003).minimize(cost)
        train_step = tf.train.MomentumOptimizer(learning_rate=0.001,
                                                momentum=0.9).minimize(cost)
        train_ops = [train_step] + r.bn_train_ops
        train_step = tf.group(*train_ops)
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('Model/train_log/trian',sess.graph)
        val_writer = tf.summary.FileWriter('Model/train_log/val')

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(r.model_var_list,max_to_keep=2)
        ckpt = tf.train.get_checkpoint_state('Model')
        #ckpt = tf.train.get_checkpoint_state(os.path.dirname('/MachineLearning/ResNet/Model'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Restore from last check point")
        else:
            print("No checkpoint found")


        batch_size = 128
        num_train = data['X_train'].shape[0]
        x_val = np.array(data['X_val'])/128.
        y_val = tf.one_hot(data['y_val'], 10).eval(session=sess)
        for step_idx in range(22180,80000):
            batch_mask = np.random.choice(num_train, batch_size)
            x_batch = np.array(data['X_train'][batch_mask])
            x_batch = image_pad_random_crop(x_batch)/128.
            y_batch = tf.one_hot(data['y_train'][batch_mask], 10).eval(session=sess)
            sess.run([train_step],feed_dict={images: x_batch, true_out: y_batch})
            if step_idx % 10 == 0:
                print step_idx
                summary = sess.run(merged, feed_dict={images: x_batch, true_out:
                                                      y_batch})
                train_writer.add_summary(summary, step_idx)
                train_writer.flush()
                if step_idx % 100 == 0:
                    summary = sess.run(merged, feed_dict={images: x_val, true_out:
                                                          y_val})
                    val_writer.add_summary(summary, step_idx)
                    val_writer.flush()
                    if step_idx % 1000 == 0:
                        saver.save(sess, 'Model/ResNet_CIFAR10')

        # test classification again, should have a higher probability about tiger
        # prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        # utils.print_prob(prob[0], './synset.txt')

        # test save
        # vgg.save_npy(sess, './test-save.npy')