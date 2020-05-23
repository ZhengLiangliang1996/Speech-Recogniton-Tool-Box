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
from keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda, TimeDistributed, Activation, Conv1D, ReLU, AveragePooling1D, Flatten
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

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

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
        self.inputs = Input(name='the_input', shape=(None, args.num_features))
        conv1       = Conv1D(32, 3, strides=1, activation='relu',
                             padding='same')(self.inputs)
        pool1       = AveragePooling1D(2, padding='same')(conv1)
        conv2       = Conv1D(64, 3, strides=1, activation='relu',padding='same')(pool1)
        pool2       = AveragePooling1D(4, padding='same')(conv2)

        # Flatten
        # flat        = Flatten()(pool2)
        dense       = Dense(64, activation='relu')(pool2)
        time_dense  = TimeDistributed(Dense(args.num_classes))(dense)
        y_pred      = Activation('softmax', name='softmax')(time_dense)
        self.model_1= Model(inputs=self.inputs, outputs = y_pred)

        self.model_1.output_length = lambda x: x

    def ctc_init(self, args):
        the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
        input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
        label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
        output_lengths = Lambda(self.model_1.output_length)(input_lengths)
        # CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                          name='ctc')([self.model_1.output, the_labels,
                          output_lengths, label_lengths])

        self.model_2 = Model(inputs=[self.model_1.input, the_labels, input_lengths,
                             label_lengths], outputs=loss_out)

        print(self.model_2.summary())

    def opt_init(self, args):
        self.opt = Adam(lr=self.args.lr, beta_1=0.9, beta_2=0.999,
                        decay = 0.01, epsilon = 10e-8)

        self.model_2.compile(loss={'ctc': lambda y_true, output: output},
                             optimizer=self.opt)
