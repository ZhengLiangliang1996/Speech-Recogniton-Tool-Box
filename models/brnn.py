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

def multi_brnn_layer(args,
                     rnn_cell,
                     inputs,
                     max_seq_length,
                     seqLengths,
                     time_major=True):

    hid_input = inputs
    for i in range(args.num_layers):
        scope = 'brnn' + str(i + 1)
        forward_cell = rnn_cell(args.hidden_size)
        backward_cell = rnn_cell(args.hidden_size)
        # tensor of shape: [max_time, batch_size, input_size]
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                           inputs=hid_input,
                                                           dtype=tf.float32,
                                                           sequence_length=seqLengths,
                                                           time_major=True,
                                                           scope=scope)
        # forward output, backward ouput
        # tensor of shape: [max_time, batch_size, input_size]
        output_fw, output_bw = outputs
        output_fb = tf.concat([output_fw, output_bw], 2)
        shape = output_fb.get_shape().as_list()
        output_fb = tf.reshape(output_fb, [shape[0], shape[1], 2, int(shape[2] / 2)])
        hidden = tf.reduce_sum(output_fb, 2)
        hidden = tf.contrib.layers.dropout(hidden, args.keep_prob, is_training = (args.mode == 'train'))

        if i != args.num_layers - 1:
            hid_input = hidden
        else:
            outputXrs = tf.reshape(hidden, [-1, args.hidden_size])
            output_list = tf.split(outputXrs, max_seq_length, 0)
            fbHrs = [tf.reshape(t, [args.batch_size, args.hidden_size]) for t in output_list]
    return fbHrs

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
<<<<<<< HEAD
            self.config = {'rnn_celltype': args.rnn_celltype,
                           'num_layers': args.num_layers,
                           'hidden_size': args.hidden_size,
                           'num_classes': args.num_classes,
                           'activation': args.activation,
                           'optimizer': args.optimizer,
                           'keep_prob': args.keep_prob,
                           'batch_size': args.batch_size,
                           'epochs': args.epochs}
=======
            self.config = {'rnn_cell_type': args.rnn_celltype,
                           'num_layer': args.num_layers,
                           'num_hidden': args.hidden_size,
                           'num_class': args.num_classes,
                           'activation': args.activation,
                           'optimizer': args.optimizer,
                           'keep prob': args.keep_prob,
                           'batch size': args.batch_size, 
                           'epoch': args.epochs}
>>>>>>> master
            self.X = tf.placeholder(tf.float32, shape = (max_seq_length, args.batch_size, args.num_features))

            #for Sparse Tensor
            self.y_indices = tf.placeholder(tf.int64, shape = (None, 2)) # y_indices eg: ([2, 3], [1, 2])
            self.y_value = tf.placeholder(tf.int64)
            self.y_shape = tf.placeholder(tf.int64)

            self.Y = tf.cast(tf.SparseTensor(self.y_indices, self.y_value, self.y_shape), tf.int32)
            self.sequence_length = tf.placeholder(tf.int32, shape=(args.batch_size, ))

            #output_drnn = multi_dynamic(args, self.X, self.rnn_cell, self.sequence_length, max_seq_length)
            output_drnn = multi_brnn_layer(args, self.rnn_cell, self.X, max_seq_length, self.sequence_length)

            # weights and biases: for the last fully connected layer
            # CTC
            with tf.name_scope('fc-layer'):
                with tf.variable_scope('fc'):
                    weights = tf.Variable(tf.truncated_normal([args.hidden_size, args.num_classes], name='weights'))
                    biases = tf.Variable(tf.zeros([args.num_classes]), name='biases')
                    # fc_W = tf.get_variable('fc_W', initializer=tf.truncated_normal([args.hidden_size, args.num_classes]))
                    # fc_b = tf.get_variable('fc_b', initializer=tf.zeros([args.num_class]))
                    logits = [tf.matmul(t, weights) + biases for t in output_drnn]

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



