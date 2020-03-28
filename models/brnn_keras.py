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
from keras.models import Input, Model
from keras.layers import Dense, Dropout, GRU, LSTM, Bidirectional, BatchNormalization, Lambda, TimeDistributed, Activation
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

class brnn_keras(object):
    def __init__(self, args, max_seq_length):
        self.max_seq_length = max_seq_length
        self.args = args
        self.model_init(args)
        self.ctc_init(args)
        self.opt_init(args)

    def model_init(self, args):
        # test
        """ Build a deep network for speech
         """
        # Main acoustic input
        self.inputs = Input(name='the_input', shape=(None, self.args.num_features))
        # Specify the layers in your network
        if self.args.rnn_celltype == 'gru':
            for i in range(self.args.num_layers):
                if i == 0:
                    bidir_rnn = Bidirectional(GRU(self.args.hidden_size,
                                                  activation=self.args.activation,
                                                  return_sequences=True,
                                                  implementation=2,
                                                  name='bidir'+str(i)),
                                                  merge_mode='concat')(self.inputs)
                    bn_rnn = BatchNormalization()(bidir_rnn)
                    dropout_rnn = Dropout(rate=self.args.keep_prob)(bn_rnn)
                else:
                    bidir_rnn = Bidirectional(GRU(self.args.hidden_size,
                                                  activation=self.args.activation,
                                                  return_sequences=True,
                                                  implementation=2,
                                                  name='bidir'+str(i)),
                                                  merge_mode='concat')(dropout_rnn)
                    bn_rnn = BatchNormalization()(bidir_rnn)
                    dropout_rnn = Dropout(rate=self.args.keep_prob)(bn_rnn)
        elif self.args.rnn_celltype == 'lstm':
            for i in range(self.args.num_layers):
                if i == 0:
                    bidir_rnn = Bidirectional(LSTM(self.args.hidden_size,
                                                  activation=self.args.activation,
                                                  return_sequences=True,
                                                  implementation=2,
                                                  name='bidir'+str(i)),
                                                  merge_mode='concat')(self.inputs)
                    bn_rnn = BatchNormalization()(bidir_rnn)
                    dropout_rnn = Dropout(rate=self.args.keep_prob)(bn_rnn)
                else:
                    bidir_rnn = Bidirectional(LSTM(self.args.hidden_size,
                                                  activation=self.args.activation,
                                                  return_sequences=True,
                                                  implementation=2,
                                                  name='bidir'+str(i)),
                                                  merge_mode='concat')(dropout_rnn)
                    bn_rnn = BatchNormalization()(bidir_rnn)
                    dropout_rnn = Dropout(rate=self.args.keep_prob)(bn_rnn)

        self.outputs = Dense(self.args.num_classes)(dropout_rnn)
        # Specify the model
        # self.model = Model(inputs= self.inputs, outputs=self.outputs)
        time_dense = TimeDistributed(Dense(self.args.num_classes))(dropout_rnn)
        # Add softmax activation layer
        y_pred = Activation('softmax', name='softmax')(time_dense)
        # Specify the model
        self.model_1 = Model(inputs=self.inputs, outputs=y_pred)
        # Specify model.output_length
        self.model_1.output_length = lambda x: x
        # print(self.model_1.summary())


    def ctc_init(self, args):
        the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
        input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
        label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
        output_lengths = Lambda(self.model_1.output_length)(input_lengths)
        # CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func,
                          output_shape=(1,),
                          name='ctc')([self.model_1.output, the_labels, output_lengths, label_lengths])
        self.model_2 = Model(inputs=[self.model_1.input, the_labels, input_lengths, label_lengths], outputs=loss_out)
        print(self.model_2.summary())

    def opt_init(self, args):
        self.opt = Adam(lr = self.args.lr, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)
        # calculate pred and errRate using edit distance
        #self.pred = tf.to_int32(tf.nn.ctc_beam_search_decoder(self.outputs, self.sequence_length, merge_repeated=False)[0][0])

        #self.errRate = tf.reduce_mean(tf.edit_distance(self.pred, self.Y, normalize=True))
        self.model_2.compile(loss={'ctc': lambda y_true, output: output}, optimizer=self.opt)

