"""
Author: Liangliang ZHENGT
Time: 2020/02/26
file: stimuli3_main.py
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
from utils.data_helper_keras_cnn import data_specification, create_batch
from utils.cha_level_helper import output_sequence
from utils.logging_helper import logging_helper
from keras.callbacks import ModelCheckpoint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser()

# Training model
parser.add_argument('--mode', type=str, default='train', help='Mode, test, train or validation')
parser.add_argument('--device', type=str, default='cpu', help='Training model with cpu or gpu')

# Training parameter
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--epochs', type=int, default=3, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
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
parser.add_argument('--pickle_name', type=str, default='morecnn2layer.pickle', help='name of the loss to be saved, extension is .pickle')
# model save name
parser.add_argument('--model_name', type=str, default='morecnn2layer.h5', help='name of the model(hdf5) to be saved, extension is h5')
# Feature level
parser.add_argument('--level', type=str, default='char', help='the feature level, could be cha, phn or seq2seq')

# Restore model
parser.add_argument('--restore', type=bool, default=False, help='Restore your model or not, default is false')


# get paser Argument
args = parser.parse_args()

logdir = args.checkpoint_dir

savedir = os.path.join(logdir, args.level, 'save')
resultdir = os.path.join(logdir, args.level, 'result')
loggingdir = os.path.join(logdir, args.level, 'logging')

if not os.path.exists(savedir):
    os.makedirs(savedir)

if not os.path.exists(resultdir):
    os.makedirs(resultdir)

if not os.path.exists(loggingdir):
    os.makedirs(loggingdir)

if args.mode == 'test':
    args.batch_size = 10
    args.epochs = 1

################################
#     SESSION
################################

logfile = os.path.join(loggingdir, str(datetime.datetime.strftime(datetime.datetime.now(),
    '%Y-%m-%d %H:%M:%S') + '.txt').replace(' ', '').replace('/', ''))

class SessionRun(object):
    def run_session(self, args):
        # get data

        ################################
        # get data and model handler
        ################################
        # training data

        data, label, _ = data_specification(args.mode, args.data_dir, 'npy')
        # 暂时设置为test 数据变化之后会进行修改
        dev_data, dev_label, _ = data_specification('test', args.data_dir,'npy')

        max_sequence_length = 0

        batch_num = len(data) // args.batch_size
        dev_batch_num = len(dev_data) // args.batch_size

        for i in data:
            max_sequence_length = max(max_sequence_length, i.shape[1])

        for i in dev_data:
            max_sequence_length = max(max_sequence_length, i.shape[1])

        max_sequence_length = 900

        # Checkpointer
        checkpointer = ModelCheckpoint(filepath=savedir+'/'+args.model_name, verbose=1)
        # Training Phase
        model = CNN(args, max_sequence_length)

        history =  model.model_2.fit_generator(generator=create_batch(args, data, max_sequence_length,label),
                                    validation_data=create_batch(args,dev_data,max_sequence_length,dev_label),
                                    steps_per_epoch=batch_num,
                                    validation_steps=dev_batch_num,
                                    epochs=args.epochs,
                                    verbose=1,
                                    callbacks=[checkpointer])
        # save model loss
        with open(savedir+'/'+args.pickle_name, 'wb') as f:
            pickle.dump(history.history, f)


if __name__ == '__main__':
    sr = SessionRun()
    sr.run_session(args)
