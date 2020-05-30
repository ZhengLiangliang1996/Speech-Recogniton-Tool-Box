#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
RNN network visualization
Inspired by: https://github.com/OverLordGoldDragon/see-rnn
"""


import argparse
import sys
import time
import datetime
import os
import _pickle as pickle
sys.path.append('..')

import numpy as np
import tensorflow as tf
from models.cnn_test import CNN
from utils.data_helper_keras import data_specification, create_batch
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train', help='Mode, test, train or validation')
parser.add_argument('--device', type=str, default='cpu', help='Training model with cpu or gpu')

# Training parameter
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--epochs', type=int, default=150, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--display_step', type=int, default=200, help='Step of displaying accuracy')
parser.add_argument('--keep_prob', type=float, default=0.5, help='Probability of dropout')

# NN architecture Parameters
parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of rnn/lstm/gru cell')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in rnn/lstm/gru cell')
parser.add_argument('--rnn_celltype', type=str, default='lstm', help='RNN cell')

parser.add_argument('--is_brnn', type=bool, default=False)
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
parser.add_argument('--activation', type=str, default='relu', help='Activation: sigmoid, tanh, relu')
parser.add_argument('--grad_clip', type=int, default=1, help='apply gradient clipping')
parser.add_argument('--layer_norm', type=bool, default=False, help='apply layer normalization')

# Training data Parameters
parser.add_argument('--num_features', type=int, default=39, help='Number of input feature')
parser.add_argument('--num_classes', type=int, default=29, help='Number of output classes')

# dir
dir_all = '../samples/cha/stimuli3/0'
parser.add_argument('--data_dir', type=str, default=dir_all, help='Data directory')
parser.add_argument('--tensorboard_log_dir', type=str, default= '../TENSORBOARD_LOG', help='TENSORBOARD_LOG directory')
parser.add_argument('--checkpoint_dir', type=str, default= '../CHECKPOINT')
# loss pickle name
parser.add_argument('--pickle_name', type=str, default='pickle_path.pickle', help='name of the loss to be saved, extension is .pickle')
# model save name
parser.add_argument('--model_name', type=str, default='Bi-lstm.h5', help='name of the model(hdf5) to be saved, extension is h5')
# Feature level
parser.add_argument('--level', type=str, default='char', help='the feature level, could be cha, phn or seq2seq')

# Restore model
parser.add_argument('--restore', type=bool, default=False, help='Restore your model or not, default is false')

args = parser.parse_args()
#data_dir = '../samples/cha/stimuli3/0'
# visualize the outlayer
def get_data(mode='train', data_dir='./PreferredStimuliData',data_suffix='npy'):
    """Return data that is used for this visualization test
       reuse the function inside the data helper, only return data with
       batchsize with 1
    """
    data, label, total_path_name =  data_specification(mode, data_dir, data_suffix)

    max_sequence_length = 0

    for i in data:
        max_sequence_length = max(max_sequence_length, i.shape[1])

    batch = create_batch(args, data, max_sequence_length, label)
    inputs, outputs = next(batch)
    #x = inputs['the_input']
    #y = inputs['the_labels'][0]
    return inputs, outputs, max_sequence_length

def make_model(max_sequence_length, model_path=''):
    # Training Phase
    model = CNN(args, max_sequence_length)

    model.model_2.load_weights(model_path)

    return model



# 768_2_LSTM
inputs, outputs,length = get_data()
x = inputs['the_input']
#print(x)

model = make_model(length, model_path='../CHECKPOINT/char/save/more_cnn_2_layer.h5')
# draw the model archtecture graph
plot_model(model.model_1, to_file='cnn_2.pdf',show_shapes=True)

