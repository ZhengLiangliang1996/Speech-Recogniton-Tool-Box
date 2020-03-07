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

sys.path.append('..')

import numpy as np
import tensorflow as tf
from models.brnn import brnn
from utils.data_helper import data_specification, create_batch
from utils.cha_level_helper import output_sequence
from utils.logging_helper import logging_helper

parser = argparse.ArgumentParser()

# Training model 
parser.add_argument('--mode', type=str, default='train', help='Mode, test, train or validation')
parser.add_argument('--device', type=str, default='cpu', help='Training model with cpu or gpu')

# Training parameter
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--epochs', type=int, default=150, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--display_step', type=int, default=200, help='Step of displaying accuracy')
parser.add_argument('--keep_prob', type=float, default=0.1, help='Probability of dropout')

# NN architecture Parameters
parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of rnn/lstm/gru cell')
parser.add_argument('--num_layers', type=int, default=7, help='Number of layers in rnn/lstm/gru cell')
parser.add_argument('--rnn_celltype', type=str, default='lstm', help='RNN cell type, rnn, gru, lstm')

# NN optimizer
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
parser.add_argument('--activation', type=str, default='relu', help='Activation: sigmoid, tanh, relu')
parser.add_argument('--grad_clip', type=int, default=1, help='apply gradient clipping')
parser.add_argument('--layer_norm', type=bool, default=False, help='apply layer normalization')

# Training data Parameters
parser.add_argument('--num_features', type=int, default=39, help='Number of input feature')
parser.add_argument('--num_classes', type=int, default=30, help='Number of output classes')

# dir
dir_all = '../samples/cha/stimuli3/0'
parser.add_argument('--data_dir', type=str, default=dir_all, help='Data directory')
parser.add_argument('--tensorboard_log_dir', type=str, default= '../TENSORBOARD_LOG', help='TENSORBOARD_LOG directory')
parser.add_argument('--checkpoint_dir', type=str, default= '../CHECKPOINT')

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

if args.mode == 'test' or args.mode == 'dev':
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
        data, label, _ = data_specification(args.mode, args.data_dir, 'npy')

        CHECKPOINT = args.checkpoint_dir
        
        max_sequence_length = 0
        for i in data:
            # print(i.shape)
            max_sequence_length = max(max_sequence_length, i.shape[1])
        
        batches = create_batch(args, data, max_sequence_length, label, args.mode)                
        model = brnn(args, max_sequence_length)

        # if args.device_name == "gpu":
        #     self.device_name = "/gpu:0"
        # else:
        #     self.device_name = "/cpu:0"
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(graph=model.graph, config=config) as sess:
            if args.restore == True:
                print('Restore from checkpoint')
                ck = tf.train.get_checkpoint_state(args.checkpoint_dir)
                if ck and ck.model_checkpoint_path:
                    model.saver.restore(sess, ck.model_checkpoint_path)
                print('Finished restoring')
            else: 
                print('Initializing All the variables')
                sess.run(model.initializer)
            
            
            all_errRate = np.zeros(args.epochs)
            all_lossRate = np.zeros(args.epochs)

            for epoch in range(args.epochs):
                start = time.time()
                cur_checkpoint_path = os.path.join(CHECKPOINT, '{:.0f}'.format(start))
                if not os.path.exists(cur_checkpoint_path):
                    os.makedirs(cur_checkpoint_path)

                if args.mode == 'train':
                    print('Epoch {} ================='.format(epoch + 1))
                

                errRate_list = np.zeros(len(batches))
                loss_list = np.zeros(len(batches))
                rand = np.random.permutation(len(batches))

                for rand_idx, batch_idx in enumerate(rand):
                    batch_data, (batch_label_indices, batch_label_value, batch_label_shape), batch_sequence_length = batches[batch_idx]
                    print("Batch_data get")
                    feeddict = {model.X: batch_data, 
                                model.y_indices: batch_label_indices,
                                model.y_value: batch_label_value, 
                                model.y_shape: batch_label_shape,
                                model.sequence_length: batch_sequence_length}

                    print("start training \n")
                    if args.mode == 'train':
                        _, batch_loss, batch_pred, batch_Y, batch_errRate = sess.run([model.optimizer, 
                                                                                    model.loss, 
                                                                                    model.pred,
                                                                                    model.Y,
                                                                                    model.errRate],
                                                                                    feed_dict = feeddict)
                        print("model train")
                        errRate_list[rand_idx] = batch_errRate
                        loss_list[rand_idx] = batch_loss
                        print('\n epoch:{},batch:{},train loss={:.3f},mean train Cha_er={:.3f}\n'.format(
                                    epoch+1, batch_idx, batch_loss, batch_errRate/args.batch_size))

                    elif args.mode == 'test':
                        batch_loss, batch_pred, batch_Y, batch_errRate = sess.run([model.loss, 
                                                                                model.pred,
                                                                                model.Y,
                                                                                model.errRate],
                                                                                feed_dict = feeddict)
                        errRate_list[rand_idx] = batch_errRate
                        loss_list[rand_idx] = batch_loss
                        print('\n epoch:{},batch:{},train loss={:.3f},mean train Cha_er={:.3f}\n'.format(
                                    epoch+1, batch_idx, batch_loss, batch_errRate/args.batch_size))

                    elif args.mode == 'val':
                        batch_loss, batch_pred, batch_Y, batch_errRate = sess.run([model.optimizer, 
                                                                                model.loss, 
                                                                                model.pred,
                                                                                model.Y,
                                                                                model.errRate],
                                                                                feed_dict = feeddict)
                        errRate_list[rand_idx] = batch_errRate
                        loss_list[rand_idx] = batch_loss
                        print('\n epoch:{},batch:{},train loss={:.3f},mean train Cha_er={:.3f}\n'.format(
                                    epoch+1, batch_idx, batch_loss, batch_errRate/args.batch_size))

                    if rand_idx % 1 == 0:
                        print('Original : ' + output_sequence(batch_Y))
                        print('Predicted: ' + output_sequence(batch_pred))
                    
                    if args.mode=='train' and ((epoch * len(rand) + rand_idx + 1) % 20 == 0 or (
                            epoch == args.epochs - 1 and rand_idx == len(rand) - 1)):
                            checkpoint_path = os.path.join(savedir, 'model.ckpt')
                            model.saver.save(sess, checkpoint_path, global_step=epoch)
                            print('Model has been saved in {}'.format(savedir))

                end = time.time()
                diff_time = end - start
                print('Epoch ' + str(epoch + 1) +'   '+ str(diff_time) + ' s')

                batches_len = len(batches)
                # save checkpoint 
                if args.mode=='train':
                    if (epoch + 1) % 1 == 0:
                        checkpoint_path = os.path.join(savedir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                        print('Model has been saved in {}'.format(savedir))
                    
                    epochER = errRate_list.sum() / (batches_len * args.batch_size)
                    epochLoss = loss_list.sum() / batches_len
                    print('Epoch', epoch + 1, 'mean train char error rate:', epochER)
                    print('Epoch', epoch + 1, 'mean train loss:', epochLoss)
                    logging_helper(model, logfile, epochLoss, epochER, epoch, diff_time, mode='config')
                    logging_helper(model, logfile, epochLoss, epochER, epoch, diff_time, mode=args.mode)
                    all_errRate[epoch] = epochER
                    all_lossRate[epoch] = epochLoss

                if args.mode == 'test' or args.mode == 'dev':
                    with open(os.path.join(resultdir, args.level + '_result.txt'), 'a') as result:
                        result.write(output_sequence(batch_Y) + '\n')
                        result.write(output_sequence(batch_pred) + '\n')
                        result.write('\n')
                    epochER = errRate_list.sum() / batches_len
                    epochLoss = loss_list.sum() / batches_len
                    print('test error rate: ', epochLoss)
                    print('test char error rate: ', epochER)
                    logging_helper(model, logfile, epochLoss, epochER, mode=args.mode)     
                    all_errRate[epoch] = epochER
                    all_lossRate[epoch] = epochLoss
            
            # save to txt file
            if args.mode == 'train' and not args.keep: 
                np.savetxt(resultdir+'/'+args.mode+'_loss.csv', all_lossRate, delimiter=',')               
                np.savetxt(resultdir+'/'+args.mode+'_char_error.csv', all_errRate, delimiter=',')            

if __name__ == '__main__':
    sr = SessionRun()
    sr.run_session(args)
