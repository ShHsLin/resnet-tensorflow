import numpy as np
import tensorflow as tf
from network.resnet import ResNet
import os
import utils.CIFAR10 as CIFAR10

def tf_image_augmentation(x_batch, pad_size=4, batch_size=128):
    images_pad = tf.image.resize_image_with_crop_or_pad(x_batch, 32 + 2 * pad_size,
                                                        32 + 2 * pad_size)
    images_pad_crop = tf.random_crop(images_pad, [batch_size, 32, 32, 3])
    images_aug = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),
                  images_pad_crop)
    return images_aug

def tf_identity(x_batch):
    return tf.identity(x_batch)

params = {'batch_size': 128,
          'data_path': '../CIFAR10/cifar-10-batches-py'}

CIFAR10 = CIFAR10.CIFAR10(params)

# data={}
# data['X_train']= CIFAR10._train_image_set
# data['y_train']= CIFAR10._train_label_set
# data['X_val']= CIFAR10._val_image_set
# data['y_val']= CIFAR10._val_label_set

config = tf.ConfigProto(allow_soft_placement=True) # , log_device_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.90
config.gpu_options.allow_growth = True

with tf.device('/gpu:0'):
    with tf.Session(config=config) as sess:
        ## Set up input image and image augmentation ##
        image_size = 32
        num_classes = 10
        batch_size = params['batch_size']
        images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        true_out = tf.placeholder(tf.int64, [None])
        aug_or_not = tf.placeholder(tf.bool)

        images_aug = tf.cond(aug_or_not,
                             lambda: tf_image_augmentation(images,
                                                           batch_size=batch_size),
                             lambda: tf_identity(images))
        # images_aug = tf.Print(images_aug, [tf.shape(images_aug)], "img_aug shape",
        #                       summarize=100)

        ## Build network ##
        r = ResNet(input_rgb=images_aug,
                   num_classes=num_classes, is_training=True)
        print('ResNet graph build, with # variables: %d' % r.get_var_count())

        ## Define Cost, Summary ##
        with tf.variable_scope('CIFAR10'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=r.logits,
                                                                  labels=true_out)
            cost = tf.reduce_mean(xent, name='xent')
            cost += r.weight_decay(0.0001) #Resnet29 should be 0.0005
            correct_prediction = tf.equal(tf.argmax(r.logits,1),
                                          true_out)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.summary.scalar('cross_entropy', cost)
            tf.summary.scalar('accuracy', accuracy)
            ## view first 5 images ##
            tf.summary.image('input image', images_aug[:5,:,:,:])

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('Model/CIFAR10/train_log/train',sess.graph)
        val_writer = tf.summary.FileWriter('Model/CIFAR10/train_log/val')

        ## Define Trainer ##
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.MomentumOptimizer(learning_rate=0.1,
                                                    momentum=0.9).minimize(cost)
#        train_ops = [train_step] + r.bn_train_ops
#        train_step = tf.group(*train_ops)
        

        ## Start Session, Initialize variables, Restore Network
        sess.run(tf.global_variables_initializer())
        ## Load from pretrain Model
        load_pretrain = False
        if load_pretrain == True:
            saver = tf.train.Saver(r.conv_var_list)
            ckpt = tf.train.get_checkpoint_state('Model')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Load pretrain model from check point")
            else:
                print("No pretrain model checkpoint found")
            saver = tf.train.Saver(r.model_var_list,max_to_keep=2)

        else:
            saver = tf.train.Saver(r.model_var_list,max_to_keep=2)
            ckpt = tf.train.get_checkpoint_state('Model/CIFAR10')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Restore from last check point")
            else:
                print("No checkpoint found")


        ## Start Training ##
        # batch_size = 128
        # num_train = data['X_train'].shape[0]
        # num_val   = data['X_val'].shape[0]
        for step_idx in range(0,30000):
            # batch_mask = np.random.choice(num_train, batch_size)
            # x_batch = np.array(data['X_train'][batch_mask])
            # y_batch = data['y_train'][batch_mask]
            x_batch, y_batch = CIFAR10.next_train_batch()
            sess.run([train_step], feed_dict={images: x_batch,
                                              true_out: y_batch,
                                              aug_or_not: r.is_training,
                                              r.bn_is_training: r.is_training})
            if step_idx % 100 == 0:
                print step_idx
                ## Train summary
                train_summary = sess.run(merged,
                                         feed_dict={images: x_batch, true_out: y_batch,
                                                    aug_or_not: r.is_training,
                                                    r.bn_is_training: r.is_training})
                train_writer.add_summary(train_summary, step_idx)
                train_writer.flush()
                ## Val summary
                r.is_training=False
                # batch_mask = np.random.choice(num_val, batch_size)
                # x_val = data['X_val'][batch_mask]
                # y_val = data['y_val'][batch_mask]
                x_val, y_val = CIFAR10.next_test_batch()
                val_summary = sess.run(merged,
                                       feed_dict={images: x_val, true_out: y_val,
                                                  aug_or_not: r.is_training,
                                                  r.bn_is_training: r.is_training})
                r.is_training=True
                val_writer.add_summary(val_summary, step_idx)
                val_writer.flush()
                ## Save Model
                if step_idx % 1000 == 0:
                    saver.save(sess, 'Model/CIFAR10/ResNet_CIFAR10')

        # test classification again, should have a higher probability about tiger
        # prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        # utils.print_prob(prob[0], './synset.txt')

        # test save
        # vgg.save_npy(sess, './test-save.npy')
