'''
Adapted mainly from
github.com/machrisaa/tensorflow-vgg/blob/master/vgg19_trainable.py
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class ResNet:
    def __init__(self, npy_path=None, is_training=True, dropout=0.5,
                 input_placeholder=None, num_classes=1000):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.is_training = is_training
        self.dropout = dropout
        self.num_classes = num_classes
        self.bn_train_ops = []
        self.build(input_placeholder)
        self.model_var_list = tf.global_variables()
        # self.conv_var_list : contain all variables before fully connected
        #                      layer. Could be save/read for transfer learning

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        # Convert RGB to BGR
        # rgb_scaled = rgb * 255.0
        # red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        # bgr = tf.concat(axis=3, values=[
        #     blue - VGG_MEAN[0],
        #     green - VGG_MEAN[1],
        #     red - VGG_MEAN[2],
        # ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        # x = bgr

        x = rgb
        #x = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), x)
        with tf.variable_scope('unit_1', reuse=None):
            x = self.conv_layer(x, 7, 3, 64, "conv1", stride_size=2)
            x = self.batch_norm(x, phase=self.is_training, scope='bn1')
            x = tf.nn.relu(x)
            x = self.max_pool(x, 'pool1', kernel_size=3)

        name = 'unit_2_1'
        x = self.bottleneck_residual(x, 64, 256, name=name, stride_size=1)
        for idx in range(2, 4):
            name = 'unit_2_%d' % idx
            x = self.bottleneck_residual(x, 256, 256, name=name)

        name = 'unit_3_1'
        x = self.bottleneck_residual(x, 256, 512, name=name)
        for idx in range(2, 5):
            name = 'unit_3_%d' % idx
            x = self.bottleneck_residual(x, 512, 512, name=name)

        name = 'unit_4_1'
        x = self.bottleneck_residual(x, 512, 1024, name=name)
        for idx in range(2, 7):
            name = 'unit_4_%d' % idx
            x = self.bottleneck_residual(x, 1024, 1024, name=name)

        name = 'unit_5_1'
        x = self.bottleneck_residual(x, 1024, 2048, name=name)
        for idx in range(2, 4):
            name = 'unit_5_%d' % idx
            x = self.bottleneck_residual(x, 2048, 2048, name=name)

        self.conv_output = tf.reduce_mean(x, [1, 2])  # global_avg_pool
        self.conv_var_list = tf.global_variables()

        self.logits = self.fc_layer(self.conv_output, 2048, self.num_classes, "fc")
        self.predict = tf.nn.softmax(self.logits, name="predict")


    def bottleneck_residual(self, x, in_channel, out_channel, name,
                            stride_size=2):
        with tf.variable_scope(name, reuse=None):
            # Identity shortcut
            if in_channel == out_channel:
                shortcut = x
                x = self.conv_layer(x, 1, in_channel, out_channel/4, "conv1")
                # conv projection shortcut
            else:
                shortcut = x
                shortcut = self.conv_layer(shortcut, 1, in_channel,
                                           out_channel, "project",
                                           stride_size=stride_size)
                shortcut = self.batch_norm(shortcut, phase=self.is_training,
                                           scope='bn0')
                x = self.conv_layer(x, 1, in_channel, out_channel/4, "conv1",
                                    stride_size=stride_size)

            x = self.batch_norm(x, phase=self.is_training, scope='bn1')
            x = tf.nn.relu(x)
            x = self.conv_layer(x, 3, out_channel/4, out_channel/4, "conv2")
            x = self.batch_norm(x, phase=self.is_training, scope='bn2')
            x = tf.nn.relu(x)
            x = self.conv_layer(x, 1, out_channel/4, out_channel, "conv3")
            x = self.batch_norm(x, phase=self.is_training, scope='bn3')
            x += shortcut
            x = tf.nn.relu(x)

        return x

    def avg_pool(self, bottom, name, kernel_size):
        return tf.nn.avg_pool(bottom, ksize=[1, kernel_size, kernel_size, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name, kernel_size=2, stride_size=2):
        return tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1],
                              strides=[1, stride_size, stride_size, 1], padding='SAME', name=name)

    def batch_norm(self, bottom, phase, scope='bn'):
        return self._batch_norm(bottom, name=scope)
    #return tf.contrib.layers.batch_norm(bottom, center=True, scale=True,
    #                                    is_training=phase, scope=scope, decay=0.9)
    # https://stackoverflow.com/questions/40879967/how-to-use-batch-normalization-correctly-in-tensorflow

    def conv_layer(self, bottom, filter_size, in_channels,
                   out_channels, name, stride_size=1):
        with tf.variable_scope(name, reuse=None):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels)
            conv = tf.nn.conv2d(bottom, filt, [1, stride_size, stride_size, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

        return bias

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name, reuse=None):
            weights, biases = self.get_fc_var(in_size, out_size)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name=""):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "weights")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name=""):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = tf.constant(self.data_dict[name][idx])
        else:
            value = initial_value  # tf.constant(initial_value)

        if self.is_training:
            var = tf.get_variable(var_name, initializer=value, trainable=True)
        else:
            var = tf.get_variable(var_name, initializer=value, trainable=False)

        self.var_dict[var.name] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def _batch_norm(self, x, name):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
            if self.is_training:
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                              initializer=tf.constant_initializer(0.0, tf.float32),
                                              trainable=False)
                moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                                  initializer=tf.constant_initializer(1.0, tf.float32),
                                                  trainable=False)

                self.bn_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self.bn_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                       initializer=tf.constant_initializer(0.0, tf.float32),
                                       trainable=False)
                variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                           initializer=tf.constant_initializer(1.0, tf.float32),
                                           trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)

            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma,
                                          0.001)
            y.set_shape(x.get_shape())
            return y


    def weight_decay(self, decay_rate=0.0001):
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weights') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(decay_rate, tf.reduce_sum(costs))

    def loss(self, labels):
        """Calculates the loss from the logits and the labels.
        Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size].

        Returns:
            loss: Loss tensor of type float.

        """
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        #self.correct = tf.nn.in_top_k(self.logits, labels, 1)
        # Return the number of true entries.
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    def training(loss, optimizer_name, learning_rate):
        """Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
            optimizer: TF optimizer object given
            loss: Loss tensor, from loss().
            learning_rate: The learning rate to use for gradient descent.

        Returns:
            train_op: The Op for training.

        """
        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)
        # Create the gradient descent optimizer with the given learning rate.
        if optimizer_name == 'Mom':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                  momentum=0.9)
        elif optimizer_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}

            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())

        return count


