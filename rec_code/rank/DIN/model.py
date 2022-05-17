#!/usr/bin/env python
"""
Created on 05 17, 2022

model: model.py

@Author: Stormzudi
"""

import tensorflow as tf
from tensorflow.python.keras.initializers import Zeros
from tensorflow.python.keras.layers import Layer, BatchNormalization
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense


# define
class Dice(Layer):
    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, inputs_shape, **kwargs):
        self.bn = BatchNormalization(self.axis, self.epsilon)
        self.alphas = self.add_weight(name='dice_bias', shape=(inputs_shape[-1],), initializer=Zeros(),
                                      dtype=tf.float32)
        super(Dice, self).build(inputs_shape)

    def call(self, inputs, **kwargs):
        bn_inputs = self.bn(inputs)
        p = tf.sigmoid(bn_inputs)
        return self.alphas * (1 - p) * inputs + p * inputs


# attention layer
class din_att(Layer):
    def __init__(self, **kwargs):
        super(din_att, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        query, keys, keys_len = inputs  # (batch, 1, embed_size), (batch, T, embed_size), (batch)中的值其实是T；
        keys_len = keys.get_shape()[1]

        querys = K.repeat_elements(query, keys_len, axis=1)
        # din中原始代码的实现方法；
        atten_input = tf.concat([querys, keys, querys - keys, querys * keys], axis=-1)  # (batch, T, 4 * embed_size)
        # 经过三层全连接层；
        dnn1 = Dense(atten_input, 80, activation=tf.nn.sigmoid, name='dnn1')
        dnn2 = Dense(dnn1, 40, activation=tf.nn.sigmoid, name='dnn2')
        dnn3 = Dense(dnn2, 1, activation=None, name='dnn3')  # (batch, T, 1)

        outputs = tf.transpose(dnn3, (0, 2, 1))  # (batch, 1, T)

        # mask
        keys_mask = tf.sequence_mask(keys_len, tf.shape(keys)[1])  # (batch, T), bool;
        keys_mask = tf.expand_dims(keys_mask, axis=1)  # (batch, 1, T)
        padding = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (batch, 1, T)
        outputs = tf.where(keys_mask, outputs, padding)  # the position of padding is set as a small num;

        # scale
        outputs = outputs / (tf.shape(keys)[-1] ** 0.5)
        outputs = tf.nn.softmax(outputs)  # (batch, 1, T)

        # weighted sum_pooling
        outputs = tf.matmul(outputs, keys)  # (batch, 1, embedding)
        return outputs