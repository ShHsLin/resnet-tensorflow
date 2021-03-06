import numpy as np
import tensorflow as tf
from utils.parse_args import parse_args
from utils.prepare_net import select_net


import os
import utils.CIFAR10 as CIFAR10

if __name__ == "__main__":
    args = parse_args()
    ckpt_dir = args.ckpt_dir  # 'Model/CIFAR10/TT_30_Adam'
    which_resnet = args.which_resnet
    bond_dim = args.bond_dim

    params={}
    params['data_path']='../CIFAR10/cifar-10-batches-py'
    # batch_size here does not matter
    params['batch_size']=64

    # CIFAR10 = read_data.CIFAR10(params)
    # data={}
    # data['X_train']= CIFAR10._train_image_set
    # data['y_train']= CIFAR10._train_label_set
    # data['X_val']= CIFAR10._val_image_set
    # data['y_val']= CIFAR10._val_label_set
    CIFAR10 = CIFAR10.CIFAR10(params)
    data={}
    data['X_val']= CIFAR10._test_image_set
    data['y_val']= CIFAR10._test_label_set

    config = tf.ConfigProto(allow_soft_placement=True) # , log_device_placement=True)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.90
    config.gpu_options.allow_growth = True

    with tf.device('/gpu:0'):
        with tf.Session(config=config) as sess:
            ## Set up input image and image augmentation ##
            image_size = 32
            num_classes = 10
            images = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
            true_out = tf.placeholder(tf.int64, [None])

            ## Build network ##
            r = select_net(which_resnet,
                           input_rgb=images,
                           num_classes=num_classes,
                           is_training=True,
                           bond_dim=bond_dim)
#             r = ResNet(input_rgb=images,
#                        num_classes=num_classes, is_training=False)
            print('ResNet graph build, with # variables: %d' % r.get_var_count())

            ## Define Cost, Summary ##
            with tf.variable_scope('CIFAR10'):
                xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=r.logits,
                                                                      labels=true_out)
                cost = tf.reduce_mean(xent, name='xent')
                cost += r.weight_decay(0.0001)
                correct_prediction = tf.equal(tf.argmax(r.logits,1),
                                              true_out)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            ## Define Trainer ##

            ## Start Session, Initialize variables, Restore Network
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(r.model_var_list,max_to_keep=2)
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Restore from last check point")
            else:
                print("No checkpoint found")


            ## Start Testing ##
            batch_size = 500
            # num_val   = data['X_val'].shape[0]

            sum_accuracy=0.
            print(data['X_val'].shape, data['y_val'].shape)
            print(data['y_val'][:10])
            num_batch = 20
            for i in range(num_batch):
                x_val = data['X_val'][batch_size*i : batch_size * (i+1)]
                y_val = data['y_val'][batch_size*i : batch_size * (i+1)]
                acc = sess.run(accuracy, feed_dict={images: x_val, true_out: y_val,
                                                    r.bn_is_training: r.is_training})
                print("%d times accuracy is %f " %(i ,acc))
                sum_accuracy += acc

            print("avg val accuracy is %f" %(sum_accuracy/num_batch))

