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
from keras.layers import Input, Dense, Lambda, TimeDistributed, Activation, Conv2D, ReLU, AveragePooling1D, MaxPooling2D, GlobalMaxPooling1D
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



class VGG(object):
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
        # Block 1
        b1_conv1    = Conv2D(32, (3,3), activation='relu', padding='same',
                             name='block1_conv1',kernel_initializer='he_normal')(self.inputs)
        b1_conv2    = Conv2D(64, (3,3), activation='relu', padding='same',
                             name='block1_conv2', kernel_initializer='he_normal')(b1_conv1)
        b1_pool     = MaxPooling2D(pool_size=(2,2),strides=None, name='block1_pool', padding='valid')(b1_conv2)

        # Block 2
        b2_conv1    = Conv2D(128, (3,3), activation='relu', padding='same',
                             name='block2_conv1', kernel_initializer='he_normal')(b1_pool)
        b2_conv2    = Conv2D(128, (3,3), activation='relu', padding='same',
                            name='block2_conv2', kernel_initializer='he_normal')(b2_conv1)
        b2_pool     = MaxPooling2D(pool_size=(2,2), strides=None, name='block2_pool', padding='valid')(b2_conv2)

        # Block 3
        b3_conv1    = Conv2D(128, (3,3), activation='relu', padding='same',
                             name='block3_conv1', kernel_initializer='he_normal')(b2_pool)
        b3_pool     = MaxPooling2D(pool_size=(2,2), strides=None, name='block3_pool', padding='valid')(b3_conv1)

        flatten     = Reshape((-1, 576))(b3_conv1)
        # Classification Block
        # cb_pool     = GlobalMaxPooling1D()(b3_pool)
        dense1      = Dense(512, activation='relu', name='fc1')(flatten)
        dense2      = Dense(256, activation='relu', name='fc2')(dense1)

        self.outputs      = Dense(args.num_classes, activation='softmax')(dense2)
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


    def opt_init(self, args):
        self.opt = Adam(lr=self.args.lr, beta_1=0.9, beta_2=0.999,
                        decay = 0.01, epsilon = 10e-8)

        self.model_2.compile(loss={'ctc': lambda y_true, output: output},
                             optimizer=self.opt)
