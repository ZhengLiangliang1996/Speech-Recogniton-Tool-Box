"""
Author: Liangliang ZHENGT
Time: 2019/12/14
file: generate_main.py
"""

import sys
import argparse
import time
import os
import glob 
import numpy as np
import tensorflow as tf
from generator_tts import generator
from utils.data_helper import data_specification, create_batch
sys.path.append('/home/liangliang/Desktop/VUB_ThirdSemester/MasterThesis/SpeechRecognitiontoolbox')

parser = argparse.ArgumentParser()
# Data Dimention
dir_all = '/home/liangliang/Desktop/VUB_ThirdSemester/MasterThesis/practice/GAN_data/'
parser.add_argument('--data_dir', type=str, default=dir_all, help='Data directory')
parser.add_argument('--tensorboard_log_dir', type=str, default= './TENSORBOARD_LOG', help='TENSORBOARD_LOG directory')
parser.add_argument('--checkpoint_dir', type=str, default= './CHECKPOINT')
parser.add_argument('--out_put_dir', type=str, default= './sample')

# Training model 
parser.add_argument('--mode', type=str, default='train', help='Mode, test, train or validation')
parser.add_argument('--device', type=str, default='cpu', help='Training model with cpu or gpu')


# get paser Argument
parser.add_argument('--condition_dimension', type=int, default=80, help='condition dimension')
parser.add_argument('--noise_term_dimension', type=int, default=128, help='condition dimension')

# get paser Argument
args = parser.parse_args()

################################
#     SESSION
################################
# TODO: 若没有这个路径则新建这个sample目录
class SessionRun(object):
    def run_session(self, args):
        # get data

        ################################
        # get data and model handler
        ################################
        dataAll, _ = data_specification(args.mode, args.data_dir, 'npy')


        # CHECKPOINT = args.checkpoint_dir
        
        # batches = create_batch(args, data, max_sequence_length, label, args.mode)                
        # model = generator(args.condition_dimension, args.noise_term_dimension)

        # if args.device_name == "gpu":
        #     self.device_name = "/gpu:0"
        # else:
        #     self.device_name = "/cpu:0"
        
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        
        for data in dataAll:
            print("-----")
            start = time.time()
            condition = tf.convert_to_tensor(data)
            condition = tf.transpose(condition)
            condition = tf.expand_dims(condition, 1)

            # iniitialize noise 
            z = tf.Variable(tf.random.normal([args.condition_dimension, args.noise_term_dimension], mean=0.0, stddev=1.0))
            tf.print(tf.shape(z))
            audio = generator(condition, z)
            tf.shape(audio)

if __name__ == '__main__':
    sr = SessionRun()
    sr.run_session(args)
