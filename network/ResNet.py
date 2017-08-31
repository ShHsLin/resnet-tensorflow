import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
from functools import reduce
import sys
sys.path.append('/usr/prakt/s032/TensorNet-TF/')
import tensornet



class ResNet:
    def __init__(self, npy_path=None, is_training=True, dropout=0.5,
                 input_rgb=None, num_classes=1000):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.is_training = is_training
        self.bn_is_training = tf.placeholder(tf.bool)
        self.dropout = dropout
        self.num_classes = num_classes
        self.bn_train_ops = []
##        with tf.variable_scope('resnet_v1_50', reuse=None):
##            self.build(input_rgb)
##
##        self.model_var_list = tf.global_variables()
        '''
        self.conv_var_list : contain all variables before fully connected
                              layer. Could be save/read for transfer learning
        '''


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
                                           out_channel, "shortcut",
                                           stride_size=stride_size)
                shortcut = self.batch_norm(shortcut, phase=self.bn_is_training,
                                           scope='shortcut/bn')
                x = self.conv_layer(x, 1, in_channel, out_channel/4, "conv1",
                                    stride_size=stride_size)

            x = self.batch_norm(x, phase=self.bn_is_training, scope='bn1')
            x = tf.nn.relu(x)
            x = self.conv_layer(x, 3, out_channel/4, out_channel/4, "conv2")
            x = self.batch_norm(x, phase=self.bn_is_training, scope='bn2')
            x = tf.nn.relu(x)
            x = self.conv_layer(x, 1, out_channel/4, out_channel, "conv3")
            x = self.batch_norm(x, phase=self.bn_is_training, scope='bn3')
            x += shortcut
            x = tf.nn.relu(x)

        return x


    def bottleneck_residual_tt(self, x, in_channel, out_channel, name,
                               stride_size=2, bond_dim=16):
        with tf.variable_scope(name, reuse=None):
            # Identity shortcut
            if in_channel == out_channel:
                shortcut = x
                x = self.conv_layer_tt(x, 1, in_channel, out_channel/4, "conv1")
                # conv projection shortcut
            else:
                shortcut = x
                shortcut = self.conv_layer_tt(shortcut, 1, in_channel,
                                              out_channel, "shortcut",
                                              stride_size=stride_size)
                shortcut = self.batch_norm(shortcut, phase=self.bn_is_training,
                                           scope='shortcut/bn')
                x = self.conv_layer_tt(x, 1, in_channel, out_channel/4, "conv1",
                                       stride_size=stride_size)

            x = self.batch_norm(x, phase=self.bn_is_training, scope='bn1')
            x = tf.nn.relu(x)
            x = self.conv_layer_tt(x, 3, out_channel/4, out_channel/4, "conv2")
            x = self.batch_norm(x, phase=self.bn_is_training, scope='bn2')
            x = tf.nn.relu(x)
            x = self.conv_layer_tt(x, 1, out_channel/4, out_channel, "conv3")
            x = self.batch_norm(x, phase=self.bn_is_training, scope='bn3')
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
#        return self._batch_norm(bottom, name=scope)
        return tf.contrib.layers.batch_norm(bottom, center=True, scale=True,
                                            is_training=phase, scope=scope,
                                            decay=0.995)
# https://stackoverflow.com/questions/40879967/how-to-use-batch-normalization-correctly-in-tensorflow

    def conv_layer(self, bottom, filter_size, in_channels,
                   out_channels, name, stride_size=1, biases=False):
        with tf.variable_scope(name, reuse=None):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels,
                                                  out_channels, biases=False)
            conv = tf.nn.conv2d(bottom, filt, [1, stride_size, stride_size, 1], padding='SAME')
            if biases == False:
                return conv
            else:
                bias = tf.nn.bias_add(conv, conv_biases)
                return bias

    def get_tt(self, filter_size, in_channels, out_channels, bond_dim=30):
        q_in_chan = np.ones(int(np.log2(in_channels))) * 2
        q_out_chan = np.ones(int(np.log2(out_channels))) * 2
        if in_channels > out_channels:
            ratio = int(in_channels/out_channels)
            extra_d = int(np.log2(ratio))
            q_out_chan = np.append(q_out_chan, np.ones(extra_d))
        elif out_channels > in_channels:
            ratio = int(out_channels/in_channels)
            extra_d = int(np.log2(ratio))
            q_in_chan = np.append(q_in_chan, np.ones(extra_d))

        tt_rank = np.ones(len(q_in_chan)+1)
        x = 1
        for i in range(len(q_in_chan)+1):
            if x < bond_dim:
                tt_rank[-i-1] = x
                x = x * q_in_chan[-i-1] * q_out_chan[-i-1]
            else:
                tt_rank[-i-1]=bond_dim

        x = filter_size * filter_size
        for i in range(len(q_in_chan)+1):
            if x < tt_rank[i]:
                tt_rank[i] = x
                x = x * q_in_chan[i] * q_out_chan[i]
            else:
                pass

        print(tt_rank)
        return (np.array(q_in_chan, dtype=np.int32),
                np.array(q_out_chan, dtype=np.int32),
                np.array(tt_rank, dtype=np.int32))


    def conv_layer_tt(self, bottom, filter_size, in_channels, out_channels,
                      name, stride_size=1, biases=False):
        quantized_in_channels, quantized_out_channels, tt_rank = \
                self.get_tt(filter_size, in_channels, out_channels)
        if biases == False:
            biases_init = None
        else:
            biases_init = tf.zeros_initializer

        return tensornet.layers.tt_conv_full(bottom,
                                             [filter_size, filter_size],
                                             quantized_in_channels,
                                             quantized_out_channels,
                                             tt_rank,
                                             strides=[stride_size,
                                                      stride_size],
                                             padding='SAME',
                                             biases_initializer=biases_init,
                                             scope=name)

        # tensornet.layers.tt_conv_full(layers[-1],
        #                               [3, 3],
        #                               np.array([4,4,4],dtype=np.int32),
        #                               np.array([4,4,4],dtype=np.int32),
        #                               np.array([16,16,16,1],dtype=np.int32),
        #                               [1, 1],
        #                               cpu_variables=cpu_variables,
        #                               biases_initializer=None,
        #                               scope='tt_conv1.2')

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name, reuse=None):
            weights, biases = self.get_fc_var(in_size, out_size)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name="",
                     biases=False):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "weights")

        if biases == False:
            return filters, None
        else:
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

    def training_step(loss, optimizer_name, learning_rate):
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
        for v in list(self.model_var_list):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        print("Total parameter, including auxiliary variables: %d\n" % count)

        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())

        return count


