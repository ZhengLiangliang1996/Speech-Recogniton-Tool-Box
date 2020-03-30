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
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Dropout, GRU, LSTM,Bidirectional,BatchNormalization, Lambda, TimeDistributed, Activation,Conv1D, ReLU
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



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class Deepspeech2(object):
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
        model = Sequential()
        # Main acoustic input

        # BatchNorm
        model.add(BatchNormalization(input_shape=(None, args.num_features),name='BN_0'))

        # 3 layer 2D Conv Layer with clippedRuLU
        model.add(Conv1D(512, 5, strides=1, activation=ReLU(20), name='Conv1D_1'))
        model.add(BatchNormalization(name="BN_1"))
        model.add(Dropout(rate=args.keep_prob))

        model.add(Conv1D(512, 5, strides=1, activation=ReLU(20),name='Conv1D_2'))
        model.add(BatchNormalization(name="BN_2"))
        model.add(Dropout(rate=args.keep_prob))

        model.add(Conv1D(512, 5, strides=1, activation=ReLU(20), name='Conv1D_3'))
        model.add(BatchNormalization(name='BN_3'))
        model.add(Dropout(rate=args.keep_prob))

        # RNN layers
        if args.rnn_celltype == 'lstm':
            model.add(Bidirectional(LSTM(args.hidden_size,
                                         activation=args.activation,
                                         return_sequences=True,
                                         name='BILSTM_1'),merge_mode='sum'))
            model.add(BatchNormalization(name='BN_4'))
            model.add(Dropout(rate=args.keep_prob))

            model.add(Bidirectional(LSTM(args.hidden_size,
                                         activation=args.activation,
                                         return_sequences=True,
                                         name='BILSTM_2'),merge_mode='sum'))
            model.add(BatchNormalization(name='BN_5'))
            model.add(Dropout(rate=args.keep_prob))

            model.add(Bidirectional(LSTM(args.hidden_size,
                                         activation=args.activation,
                                         return_sequences=True,
                                         name='BILSTM_3'),merge_mode='sum'))
            model.add(BatchNormalization(name='BN_6'))
            model.add(Dropout(rate=args.keep_prob))

            model.add(Bidirectional(LSTM(args.hidden_size,
                                         activation=args.activation,
                                         return_sequences=True,
                                         name='BILSTM_4'),merge_mode='sum'))
            model.add(BatchNormalization(name='BN_7'))
            model.add(Dropout(rate=args.keep_prob))

            model.add(Bidirectional(LSTM(args.hidden_size,
                                         activation=args.activation,
                                         return_sequences=True,
                                         name='BILSTM_5'),merge_mode='sum'))
            model.add(BatchNormalization(name='BN_8'))
            model.add(Dropout(rate=args.keep_prob))

            model.add(Bidirectional(LSTM(args.hidden_size,
                                         activation=args.activation,
                                         return_sequences=True,
                                         name='BILSTM_6'),merge_mode='sum'))
            model.add(BatchNormalization(name='BN_9'))
            model.add(Dropout(rate=args.keep_prob))

            model.add(Bidirectional(LSTM(args.hidden_size,
                                         activation=args.activation,
                                         return_sequences=True,
                                         name='BILSTM_7'),merge_mode='sum'))
            model.add(BatchNormalization(name='BN_10'))
            model.add(Dropout(rate=args.keep_prob))

            # Fully Connected Layer
            model.add(TimeDistributed(Dense(args.num_size_fc, activation=ReLU(20), name='FC_1')))
            model.add(TimeDistributed(Dense(args.num_classes,activation='softmax', name='Y_pred')))

        return model
    def ctc_init(self, args):
        y_pred = self.model.outputs[0]
        model_input = self.model.inputs[0]

        labels = Input(name='the_labels', shape=[None,], dtype='int32')
        input_length = Input(name='input_length', shape=[1], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')

        loss_out = Lambda(ctc_lambda_func, name='ctc')([y_pred, labels, input_length, label_length])

        self.model_2 = Model(inputs=[model_input, labels, input_length, label_length], outputs=loss_out)
        print(self.model_2.summary())

    def opt_init(self, args):
        self.opt = Adam(lr = self.args.lr, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)

        self.model_2.compile(loss={'ctc': lambda y_true, output: output}, optimizer=self.opt)
