import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
from functools import reduce
from ResNet import ResNet


class resnet_v1_29(ResNet):
    def __init__(self, npy_path=None, is_training=True, dropout=0.5,
                 input_rgb=None, num_classes=1000):
        ResNet.__init__(self, npy_path, is_training, dropout,
                        input_rgb, num_classes)

        with tf.variable_scope('resnet_v1_29', reuse=None):
            self.build(input_rgb)

        self.model_var_list = tf.global_variables()
        # self.conv_var_list : contain all variables before fully connected
        #                      layer. Could be save/read for transfer learning

    def build(self, rgb):
        """
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
        num_units = [3, 3, 3]

        with tf.variable_scope('block0', reuse=None):
            x = self.conv_layer(x, 3, 3, 64, "conv1", stride_size=1, biases=False)
            x = self.batch_norm(x, phase=self.bn_is_training, scope='bn1')
            x = tf.nn.relu(x)
            # x = self.max_pool(x, 'pool1', kernel_size=3)

        with tf.variable_scope('block1', reuse=None):
            name = 'unit_1'
            x = self.bottleneck_residual(x, 64, 256, name=name, stride_size=1)
            for idx in range(2, num_units[0]+1):
                name = 'unit_%d' % idx
                x = self.bottleneck_residual(x, 256, 256, name=name)

        with tf.variable_scope('block2', reuse=None):
            name = 'unit_1'
            x = self.bottleneck_residual(x, 256, 512, name=name)
            for idx in range(2, num_units[1]+1):
                name = 'unit_%d' % idx
                x = self.bottleneck_residual(x, 512, 512, name=name)

        with tf.variable_scope('block3', reuse=None):
            name = 'unit_1'
            x = self.bottleneck_residual(x, 512, 1024, name=name)
            for idx in range(2, num_units[2]+1):
                name = 'unit_%d' % idx
                x = self.bottleneck_residual(x, 1024, 1024, name=name)

        self.conv_output = tf.reduce_mean(x, [1, 2])  # global_avg_pool
        self.conv_var_list = tf.global_variables()

        self.logits = self.fc_layer(self.conv_output, 1024, self.num_classes,
                                    "logits")
        self.predict = tf.nn.softmax(self.logits, name="predict")
        return


class resnet_v1_29_tt(ResNet):
    def __init__(self, npy_path=None, is_training=True, dropout=0.5,
                 input_rgb=None, num_classes=1000, bond_dim=30):
        ResNet.__init__(self, npy_path, is_training, dropout,
                        input_rgb, num_classes, bond_dim)

        with tf.variable_scope('resnet_v1_29_tt', reuse=None):
            self.build(input_rgb)

        self.model_var_list = tf.global_variables()
        # self.conv_var_list : contain all variables before fully connected
        #                      layer. Could be save/read for transfer learning

    def build(self, rgb):
        """
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        x = rgb
        num_units = [3, 3, 3]

        with tf.variable_scope('block0', reuse=None):
            x = self.conv_layer(x, 3, 3, 64, "conv1", stride_size=1, biases=False)
            x = self.batch_norm(x, phase=self.bn_is_training, scope='bn1')
            x = tf.nn.relu(x)
            # x = self.max_pool(x, 'pool1', kernel_size=3)

        with tf.variable_scope('block1', reuse=None):
            name = 'unit_1'
            x = self.bottleneck_residual_tt(x, 64, 256, name=name, stride_size=1)
            for idx in range(2, num_units[0]+1):
                name = 'unit_%d' % idx
                x = self.bottleneck_residual_tt(x, 256, 256, name=name)

        with tf.variable_scope('block2', reuse=None):
            name = 'unit_1'
            x = self.bottleneck_residual_tt(x, 256, 512, name=name)
            for idx in range(2, num_units[1]+1):
                name = 'unit_%d' % idx
                x = self.bottleneck_residual_tt(x, 512, 512, name=name)

        with tf.variable_scope('block3', reuse=None):
            name = 'unit_1'
            x = self.bottleneck_residual_tt(x, 512, 1024, name=name)
            for idx in range(2, num_units[2]+1):
                name = 'unit_%d' % idx
                x = self.bottleneck_residual_tt(x, 1024, 1024, name=name)

        self.conv_output = tf.reduce_mean(x, [1, 2])  # global_avg_pool
        self.conv_var_list = tf.global_variables()

        self.logits = self.fc_layer(self.conv_output, 1024, self.num_classes,
                                    "logits")
        self.predict = tf.nn.softmax(self.logits, name="predict")
        return





class resnet_v1_50(ResNet):
    def __init__(self, npy_path=None, is_training=True, dropout=0.5,
                 input_rgb=None, num_classes=1000):
        ResNet.__init__(self, npy_path, is_training, dropout,
                        input_rgb, num_classes)

        with tf.variable_scope('resnet_v1_50', reuse=None):
            self.build(input_rgb)

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
        with tf.variable_scope('block0', reuse=None):
            x = self.conv_layer(x, 7, 3, 64, "conv1", stride_size=2, biases=False)
            x = self.batch_norm(x, phase=self.bn_is_training, scope='bn1')
            x = tf.nn.relu(x)
            x = self.max_pool(x, 'pool1', kernel_size=3)

        with tf.variable_scope('block1', reuse=None):
            name = 'unit_1'
            x = self.bottleneck_residual(x, 64, 256, name=name, stride_size=1)
            for idx in range(2, 4):
                name = 'unit_%d' % idx
                x = self.bottleneck_residual(x, 256, 256, name=name)

        with tf.variable_scope('block2', reuse=None):
            name = 'unit_1'
            x = self.bottleneck_residual(x, 256, 512, name=name)
            for idx in range(2, 5):
                name = 'unit_%d' % idx
                x = self.bottleneck_residual(x, 512, 512, name=name)

        with tf.variable_scope('block3', reuse=None):
            name = 'unit_1'
            x = self.bottleneck_residual(x, 512, 1024, name=name)
            for idx in range(2, 7):
                name = 'unit_%d' % idx
                x = self.bottleneck_residual(x, 1024, 1024, name=name)

        with tf.variable_scope('block4', reuse=None):
            name = 'unit_1'
            x = self.bottleneck_residual(x, 1024, 2048, name=name)
            for idx in range(2, 4):
                name = 'unit_%d' % idx
                x = self.bottleneck_residual(x, 2048, 2048, name=name)

        self.conv_output = tf.reduce_mean(x, [1, 2])  # global_avg_pool
        self.conv_var_list = tf.global_variables()

        self.logits = self.fc_layer(self.conv_output, 2048, self.num_classes,
                                    "logits")
        self.predict = tf.nn.softmax(self.logits, name="predict")




