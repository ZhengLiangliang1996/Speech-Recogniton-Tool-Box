#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
File Description
"""
import sys
import time
import os

sys.path.append('..')

from __future__ import print_function
import sys
import argparse
from keras import backend as K
from utils.cha_level_helper import output_sequence
import numpy as np

#TODO: still need to be tested
def get_predictions_then_print(data, label, mode, model, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): dataset index
        mode: which will get the dataset
        model: model will be used
        model_path (str): model checkpoint
    """
    data_len = len(data)

    for i in range(data_len):
        # Obtain and decode the acoustic model's predictions
        model.load_weights(model_path)
        prediction = model.predict(data[i])
        output_length = [model.output_length(data[i].shape[1])]
        #why + 1?
        pred_ints = (K.eval(K.ctc_decode(
                    prediction, output_length)[0][0])).flatten().tolist()

        # Play the audio file, and display the true and predicted transcriptions
        print('-'*80)
        print('Ground Truth:\n' + '\n' + output_sequence(label[i]))
        print('-'*80)
        print('Predicted seq:\n' + '\n' + ''.join(output_sequence(pred_ints)))
        print('-'*80)
