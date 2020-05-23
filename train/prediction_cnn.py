#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date: 21 Apr 2020
This file is for text prediction
"""

import argparse
import sys
import time
import datetime
import os
import _pickle as pickle
from glob import glob
sys.path.append('..')

import numpy as np
import tensorflow as tf
from models.cnn_test import CNN
from models.vgg import VGG
from utils.data_helper_keras_cnn import data_specification, create_batch
from utils.cha_level_helper import int_sequence_to_text,int_sequence_to_text_test
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# Training model
parser.add_argument('--mode', type=str, default='train', help='Mode, test, train or validation')
parser.add_argument('--device', type=str, default='cpu', help='Training model with cpu or gpu')

# Training parameter
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--epochs', type=int, default=150, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--display_step', type=int, default=200, help='Step of displaying accuracy')
parser.add_argument('--keep_prob', type=float, default=0.5, help='Probability of dropout')

# NN architecture Parameters
parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of rnn/lstm/gru cell')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in rnn/lstm/gru cell')
parser.add_argument('--rnn_celltype', type=str, default='lstm', help='RNN cell type, gru, lstm')
# fc layer num
parser.add_argument('--num_size_fc', type=int, default=1024, help='number of unit in fc layer ')

# NN optimizer
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
parser.add_argument('--pickle_name', type=str, default='cnn2layer.pickle', help='name of the loss to be saved, extension is .pickle')
# model save name
parser.add_argument('--model_name', type=str, default='cnn2layer.h5', help='name of the model(hdf5) to be saved, extension is h5')
# Feature level
parser.add_argument('--level', type=str, default='char', help='the feature level, could be cha, phn or seq2seq')

# Restore model
parser.add_argument('--restore', type=bool, default=False, help='Restore your model or not, default is false')

args = parser.parse_args()


# CTC decoder
import os
import tensorflow as tf
import numpy as np
from keras import backend as K

# CTC_decoder:
def decode_ctc(args, num_result):
    result = num_result[:, :, :]
    # transpose, since the batched data in my case is max_length, batch_size, num_feature
    #result = result.transpose((1,0,2))
    in_len = np.zeros((args.batch_size), dtype = np.int32)
    in_len[0] = result.shape[1]
    r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
    r1 = K.get_value(r[0][0])
    r1 = r1[0]
    text = []
    for i in r1:
        text.append(i)
    return text

def output_matrix(m):
    plt.matshow(m.T)
    plt.colorbar()
    plt.show()

def prediction(args, number, data_dir='../samples/cha/stimuli3/0', model_path="", mode='train', data_suffix = 'npy'):
    """ Get the ground truth and predicted sequence
        number: number of data in the batch wants to be visualized
        mode: train or validation or test, which will indicate the folder to be opened
        dir: the directory of the mfcc data
        model_path: the model checkpoint, HDF5 file
        data_suffix: the data suffix of those audio file inside the data_dir
    """
    # get data by index
    data, label, total_path_name =  data_specification(mode, data_dir, data_suffix)

    max_sequence_length = 0


    for i in data:
        max_sequence_length = max(max_sequence_length, i.shape[1])

    # Training Phase
    model = CNN(args, max_sequence_length)

    model.model_2.load_weights(model_path)

    batch = create_batch(args, data, max_sequence_length, label)
    for i in range(number):
        inputs, outputs = next(batch)

        x = inputs['the_input']
        y = inputs['the_labels'][0]


        prediction = model.model_1.predict(x)
        print(np.shape(prediction))
        pred_ints = decode_ctc(args, prediction)

        model_path = model_path.replace("../CHECKPOINT/char/save/", "")
        print('model '+ model_path)
        print('-'*80)
        y = y.astype(int)
        if "test" not in model_path:
            print('GroundTruth speech:\n' + '\n' + int_sequence_to_text(y))
        else:
            print('GroundTruth speech:\n' + '\n' + int_sequence_to_text_test(y))

        print('-'*80)
        print(pred_ints)
        if "test" not in model_path:
            print('Predicted speech:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
        else:
            print('Predicted speech:\n' + '\n' + ''.join(int_sequence_to_text_test(pred_ints)))


        output_matrix(prediction[0])
def run_prediction(args, model_path):
    all_models = sorted(glob(model_path+"cnn2layer.h5"))

    print(all_models)
    model_names = [item[:] for item in all_models]
    data_dir='../sample_test/cha/stimuli3/0'
    print(args)
    for i in model_names:
        prediction(args, number=2, data_dir=data_dir,
                   model_path=i)


run_prediction(args, '../CHECKPOINT/char/save/')

def word_error_rate():
    pass
