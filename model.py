# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model
   Description :
   Author :       iffly
   date：          4/23/18
-------------------------------------------------
   Change Activity:
                   4/23/18:
-------------------------------------------------
"""
import tensorflow as tf


class C3d(object):
    def __init__(self, num_class=20, keep_prob=0.5, wd=0.00005, frame_num=16, size_w=112, size_h=112, chanel_num=3,
                 weight_init=tf.contrib.layers.xavier_initializer()):
        self.num_class = num_class
        self.keep_prob = keep_prob
        self.frame_num = frame_num
        self.size_w = size_w
        self.size_h = size_h
        self.chanel_num = chanel_num
        self.wd = wd
        self.weight_init = weight_init

    def _variable(self, name, shape, initializer):
        with tf.device("/cpu:0"):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        var = self._variable(name, shape, self.weight_init)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd)
            tf.add_to_collection('weight_decay_loss', weight_decay)
        return var

    def conv3d(self, name, l_input, w, b):
        with tf.variable_scope(name) as scope:
            return tf.nn.bias_add(
                tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
                b
            )

    def max_pool(self, name, l_input, k):
        return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

    def build_weight(self):
        with tf.variable_scope('var_name') as var_scope:
            wd = self.wd
            self.weights = {
                'wc1': self._variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, wd),
                'wc2': self._variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, wd),
                'wc3a': self._variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, wd),
                'wc3b': self._variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, wd),
                'wc4a': self._variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, wd),
                'wc4b': self._variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, wd),
                'wc5a': self._variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, wd),
                'wc5b': self._variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, wd),
                'wd1': self._variable_with_weight_decay('wd1', [8192, 4096], 0.04, wd),
                'wd2': self._variable_with_weight_decay('wd2', [4096, 4096], 0.04, wd),
                'out': self._variable_with_weight_decay('wout', [4096, self.num_class], 0.04, wd)
            }
            self.biases = {
                'bc1': self._variable_with_weight_decay('bc1', [64], 0.04, None),
                'bc2': self._variable_with_weight_decay('bc2', [128], 0.04, None),
                'bc3a': self._variable_with_weight_decay('bc3a', [256], 0.04, None),
                'bc3b': self._variable_with_weight_decay('bc3b', [256], 0.04, None),
                'bc4a': self._variable_with_weight_decay('bc4a', [512], 0.04, None),
                'bc4b': self._variable_with_weight_decay('bc4b', [512], 0.04, None),
                'bc5a': self._variable_with_weight_decay('bc5a', [512], 0.04, None),
                'bc5b': self._variable_with_weight_decay('bc5b', [512], 0.04, None),
                'bd1': self._variable_with_weight_decay('bd1', [4096], 0.04, None),
                'bd2': self._variable_with_weight_decay('bd2', [4096], 0.04, None),
                'out': self._variable_with_weight_decay('bout', [self.num_class], 0.04, None),
            }

    def getweights(self):
        return self.weights

    def getbiases(self):
        return self.biases

    def build_model(self, input=None, weights=None):
        if weights:
            self.weights = weights
        else:
            self.build_weight()
        if not input is None:
            self.input = input
        else:
            self.input = tf.placeholder(tf.float32, shape=(None,
                                                           self.frame_num,
                                                           self.size_w,
                                                           self.size_w,
                                                           self.chanel_num))
        _weights = self.weights
        _biases = self.biases
        conv3d = self.conv3d
        max_pool = self.max_pool
        _dropout = self.keep_prob
        # Convolution Layer
        net = conv3d('conv1', self.input, _weights['wc1'], _biases['bc1'])
        net = tf.nn.relu(net, 'relu1')
        net = max_pool('pool1', net, k=1)

        # Convolution Layer
        net = conv3d('conv2', net, _weights['wc2'], _biases['bc2'])
        net = tf.nn.relu(net, 'relu2')
        net = max_pool('pool2', net, k=2)

        # Convolution Layer
        net = conv3d('conv3a', net, _weights['wc3a'], _biases['bc3a'])
        net = tf.nn.relu(net, 'relu3a')
        net = conv3d('conv3b', net, _weights['wc3b'], _biases['bc3b'])
        net = tf.nn.relu(net, 'relu3b')
        net = max_pool('pool3', net, k=2)

        # Convolution Layer
        net = conv3d('conv4a', net, _weights['wc4a'], _biases['bc4a'])
        net = tf.nn.relu(net, 'relu4a')
        net = conv3d('conv4b', net, _weights['wc4b'], _biases['bc4b'])
        net = tf.nn.relu(net, 'relu4b')
        net = max_pool('pool4', net, k=2)

        # Convolution Layer
        net = conv3d('conv5a', net, _weights['wc5a'], _biases['bc5a'])
        net = tf.nn.relu(net, 'relu5a')
        net = conv3d('conv5b', net, _weights['wc5b'], _biases['bc5b'])
        net = tf.nn.relu(net, 'relu5b')
        net = max_pool('pool5', net, k=2)

        # Fully connected layer
        net = tf.transpose(net, perm=[0, 1, 4, 2, 3])
        net = tf.reshape(net, [-1, _weights['wd1'].get_shape().as_list()[
            0]])  # Reshape conv3 output to fit dense layer input
        net = tf.nn.bias_add(tf.matmul(net, _weights['wd1']), _biases['bd1'])

        net = tf.nn.relu(net, name='fc1')  # Relu activation
        net = tf.nn.dropout(net, _dropout)

        net = tf.nn.relu(tf.nn.bias_add(tf.matmul(net, _weights['wd2']), _biases['bd2']), name='fc2')  # Relu activation
        net = tf.nn.dropout(net, _dropout)

        # Output: class prediction
        net = tf.nn.bias_add(tf.matmul(net, _weights['out']), _biases['out'], name='out')

        return net
