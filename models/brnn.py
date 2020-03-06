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

sys.path.append('..')

from tensorflow.contrib import rnn
from utils.rnn_cell_helper import RNN_cell

# Multi dynamic brnn model
# Scope PART LEFT
# def multi_dynamic(args, inputs, rnn_cell, sequence_length, max_seq_length):
#     print(inputs)
#     hidden_inputs = inputs
#     for i in range(args.num_layers):
#         # Forward direction cell
#         rnn_fw_cell = rnn_cell(hidden_inputs, args.activation)
#         # Backward direction cell
#         rnn_bwd_cell = rnn_cell(hidden_inputs, args.activation)
#         (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell,
#                                         rnn_bwd_cell, 
#                                         inputs = hidden_inputs,
#                                         sequence_length=sequence_length,
#                                         dtype=tf.float32,
#                                         time_major=True,
#                                         scope='multi_dynamic_{}'.format(i))

#         print(tf.shape(output_fw))
#         hidden_outputs = output_fw + output_bw 
#         print(tf.shape(hidden_outputs))
#         hidden_outputs = tf.contrib.layers.dropout(hidden_outputs, keep_prob = args.dropout, is_training =('train' == args.mode))

#         # 
#         if i < args.num_layers - 1:
#             hidden_inputs = hidden_outputs
#         else:
#             # 分成max_seq_length等分
#             output_splited = tf.split(hidden_outputs, max_seq_length, axis=0)
#             # 遵循self.X的shape格式
#             output_drnn = [tf.reshape(t, shape=(args.batch_size, args.hidden_size)) for t in output_splited]
    
#     return output_drnn


def brnn_layer(fw_cell, bw_cell, inputs, seq_lengths, scope=None):
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                                bw_cell,
                                                                inputs=inputs,
                                                                dtype=tf.float32,
                                                                sequence_length=seq_lengths,
                                                                time_major=True,
                                                                scope=scope)

    brnn_combined_outputs = output_fw + output_bw
    return brnn_combined_outputs

def multi_brnn_layer(args, rnn_cell, inputs, seq_lengths, num_layers, is_training, use_dropout=True, keep_prob=0.5):
    inner_outputs = inputs
    for n in range(num_layers):
        forward_cell = rnn_cell(args.hidden_size)
        backward_cell = rnn_cell(args.hidden_size)
        inner_outputs = brnn_layer(forward_cell, backward_cell, inner_outputs, seq_lengths, 'brnn_{}'.format(n))
        if use_dropout:
            inner_outputs = tf.contrib.layers.dropout(inner_outputs, keep_prob=keep_prob, is_training=is_training)

    return inner_outputs

class brnn(object):
    def __init__(self, args, max_seq_length):
        self.args = args
        self.max_seq_length = max_seq_length
        mid = RNN_cell()
        self.rnn_cell = mid.make_cell(args.rnn_celltype, args.layer_norm)
        self.graph_run(args, max_seq_length)
        
    # def graph_run(self, args):
    def graph_run(self, args, max_seq_length):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.config = {'rnn_cell_type': args.rnn_celltype,
                           'num_layer': args.num_layers,
                           'num_hidden': args.hidden_size,
                           'num_class': args.num_classes,
                           'activation': args.activation,
                           'optimizer': args.optimizer,
                           'keep prob': args.keep_prob,
                           'batch size': args.batch_size, 
                           'epoch': args.epochs}
            self.X = tf.placeholder(tf.float32, shape = (max_seq_length, args.batch_size, args.num_features))
            
            #for Sparse Tensor
            self.y_indices = tf.placeholder(tf.int64, shape = (None, 2)) # y_indices eg: ([2, 3], [1, 2])
            self.y_value = tf.placeholder(tf.int64)
            self.y_shape = tf.placeholder(tf.int64)

            self.Y = tf.cast(tf.SparseTensor(self.y_indices, self.y_value, self.y_shape), tf.int32)
            self.sequence_length = tf.placeholder(tf.int32, shape=(args.batch_size, ))

            #output_drnn = multi_dynamic(args, self.X, self.rnn_cell, self.sequence_length, max_seq_length)
            output_drnn = multi_brnn_layer(args, self.rnn_cell, self.X, self.sequence_length, args.num_layers, True, keep_prob=0.5)
            output_drnn = [tf.reshape(t, shape=(args.batch_size, args.hidden_size)) for t in tf.split(output_drnn,max_seq_length,0)]
            # weights and biases: for the last fully connected layer
            # CTC
            fc_W = tf.get_variable('fc_W', initializer=tf.truncated_normal([args.hidden_size, args.num_classes]))
            fc_b = tf.get_variable('fc_b', initializer=tf.truncated_normal([args.num_classes]))

            logits = [tf.matmul(t, fc_W) + fc_b for t in output_drnn]
            logits_stack = tf.stack(logits, axis = 0)
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.Y, logits_stack, self.sequence_length))

            self.var_trainable_op = tf.trainable_variables()
            
            if args.grad_clip == 1:
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.var_trainable_op), args.grad_clip)
                self.optimizer = tf.train.AdamOptimizer(args.lr).apply_gradients(zip(grads, self.var_trainable_op))
            else:
                self.optimizer = tf.train.AdamOptimizer(args.lr).minimize(self.loss)

            self.pred = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits_stack, self.sequence_length, merge_repeated=False)[0][0])

            self.errRate = tf.reduce_mean(tf.edit_distance(self.pred, self.Y, normalize=True))

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
            self.initializer = tf.global_variables_initializer()
            


