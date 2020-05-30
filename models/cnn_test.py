#! /usr/bin/env python
"""
Author: Liangliang ZHENGT
Time: 2019/11/6
model: BRNN
Remarks: Some of the part, #Training Parameters # Training data Parameters # Import data
         need to be modified to arg using argparse
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
import argparse
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda, TimeDistributed, Activation, Conv2D, ReLU, AveragePooling1D, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Reshape
from keras.optimizers import Adam
from keras import backend as K


sys.path.append('..')

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
tf.keras.backend.clear_session()  # For easy reset of notebook state.

config_proto = tf.compat.v1.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
session = tf.Session(config=config_proto)
set_session(session)


def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# def ctc_lambda_func(args):
    # y_pred, labels, input_length, label_length = args
    # return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def conv2d(size):
    return Conv2D(size, (3,3), use_bias=True, activation='relu',
        padding='same', kernel_initializer='he_normal')


def norm(x):
    return BatchNormalization(axis=-1)(x)


def maxpool(x):
    return MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(x)


def dense(units, activation="relu"):
    return Dense(units, activation=activation, use_bias=True,
        kernel_initializer='he_normal')


# x.shape=(none, none, none)
# output.shape = (1/2, 1/2, 1/2)
def cnn_cell(size, x, pool=True):
    x = norm(conv2d(size)(x))
    x = norm(conv2d(size)(x))
    if pool:
        x = maxpool(x)
    return x

class CNN(object):
    def __init__(self, args, max_seq_length):
        self.max_seq_length = max_seq_length
        self.args = args
        self.model = self.model_init(args)
        self.ctc_init(args)
        self.opt_init(args)

    # TODO: the keep_prob should be different
    def model_init(self, args):
        # test
        """ Build a deep network for speech
        """
        self.inputs = Input(name='the_input', shape=(None, args.num_features, 1))
        self.h1 = cnn_cell(32, self.inputs)
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2)
        self.h4 = cnn_cell(128, self.h3, pool=False)
        self.h5 = cnn_cell(128, self.h4, pool=False)
        # 39 / 8 * 256 = 3200
        self.h6 = Reshape((-1, 512))(self.h3)
        #self.h6 = Dropout(0.2)(self.h6)
        self.h66 = dense(512)(self.h6)
        self.h7 = dense(256)(self.h6)
        self.h7 = Dropout(0.2)(self.h7)
        self.outputs = dense(args.num_classes, activation='softmax')(self.h7)
        self.model_1= Model(inputs=self.inputs, outputs = self.outputs)
        self.model_1.summary()

    def ctc_init(self, args):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')\
            ([self.labels, self.outputs, self.input_length, self.label_length])
        self.model_2 = Model(inputs=[self.labels, self.inputs,
            self.input_length, self.label_length], outputs=self.loss_out)

    # def ctc_init(self, args):
        # the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
        # input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
        # label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
        # output_lengths = Lambda(self.model_1.output_length)(input_lengths)
        # # CTC loss is implemented in a lambda layer
        # loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                          # name='ctc')([self.model_1.output, the_labels,
                          # output_lengths, label_lengths])

        # self.model_2 = Model(inputs=[self.model_1.input, the_labels, input_lengths,
                             # label_lengths], outputs=loss_out)

        # print(self.model_2.summary())

    def opt_init(self, args):
        self.opt = Adam(lr=self.args.lr, beta_1=0.9, beta_2=0.999,
                        decay = 0.01, epsilon = 10e-8)

        self.model_2.compile(loss={'ctc': lambda y_true, output: output},
                             optimizer=self.opt)
