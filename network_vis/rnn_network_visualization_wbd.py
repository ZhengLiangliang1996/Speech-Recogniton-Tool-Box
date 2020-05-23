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
from models.vis_model.brnn_keras_wbd import brnn_keras
from utils.data_helper_keras import data_specification, create_batch
import tensorflow.keras.backend as K
from util import show_features_2D, selectunits
from util import show_scatter_density,show_box_plot
from visuals_rnn import rnn_histogram, rnn_heatmap
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
parser.add_argument('--rnn_celltype', type=str, default='gru', help='RNN cell')

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
    model = brnn_keras(args, max_sequence_length)

    model.model_2.load_weights(model_path)

    return model

def get_layer(model, layer_idx=None, layer_name=None):
    """get model layer by index
    If get layer by layer_name, if there are multiple layers returned,
    return the earliest one
    """
    #validate_args(layer_name, layer_idx, layer=None)
    if layer_idx is not None:
        return model.layers[layer_idx]

    layer = [layer for layer in model.layers if layer_name in layer.name]

    if len(layer) > 1:
        print(warn_str + "multiple matching layer names found; "
              + "picking earliest")
    elif len(layer) == 0:
        raise Exception("no layers found w/ names matching "
                        + "substring: '%s'" % layer_name)
    return layer[0]

def get_layer_outputs(model, input_data, layer_name=None, layer_idx=None,
                      learning_phase=0):
    # model initialization
    tf.keras.backend.get_session().run(tf.global_variables_initializer())
    layer = get_layer(model, layer_name = layer_name, layer_idx = layer_idx)
    layers_fn = K.function([model.input], [layer.output])

    return layers_fn([input_data, learning_phase])[0]

def output_visualize(model, input_data = None, layer_idx = None, layer_name = None, unit_choose = 32, n_rows=1):
    """visualize the detail inside rnn and brnn
        Arguments:
            model: the model loaded from chekcpoint
            layer_idx: the index of the layer inside the network
            layer_name: the layer name defined inside the network
            unit_choose: the number of units needed to be visualized, default is 32
    """
    # get the output
    outs = get_layer_outputs(model, input_data, layer_name=layer_name,
                             layer_idx=layer_idx)

    print(np.shape(outs))
    # show the data in graph
    if np.shape(outs)[2] > unit_choose:
        outs1 = selectunits(outs, units = unit_choose, is_bilstm=args.is_brnn)
    if np.shape(outs)[2] < unit_choose:
        unit_choose = np.shape(outs)[2]

    # (1, 108, 29) batch number, max_size, unit
    # show_scatter_density(outs, units = unit_choose)
    # 1D of features, only select one batch
    # show_features_1D(outs[:1], n_rows=n_rows)
    if np.shape(outs)[2] > unit_choose:
        show_box_plot(outs1, units = unit_choose)
    else:
        show_box_plot(data=outs, units = unit_choose)
    show_features_2D(outs, n_rows=4, norm=(-.1, .1))

def main():
    # get data
    inputs, outputs,length = get_data()
    # get input of the data
    x = inputs['the_input']
    # make model
    model = make_model(length, model_path='../CHECKPOINT/char/save/wbd_768_unit_2_layer_GRU.h5')
    # draw the model archtecture graph
    # plot_model(model.model_2, to_file='model.png',show_shapes=True)

    # vializatie the output, 1D and 2D
#     output_visualize(model.model_2, input_data = x, layer_name='lstm1',
                     # n_rows = 4, unit_choose = 64)

    # visualize the inner RNN weights by kernel and get
    rnn_histogram(model.model_2, 'gru1', equate_axes=False)
    rnn_heatmap(model.model_2, 'gru')
if __name__ == "__main__":
    main()

