"""
Author: Liangliang ZHENGT
Time: 2019/11/6
file: main.py
"""


import argparse
import sys
import time
<<<<<<< HEAD
import os
=======
import os
>>>>>>> master

sys.path.append('..')

import numpy as np
import tensorflow as tf
from models.brnn import brnn
<<<<<<< HEAD
from models.brnn_keras import brnnkeras
=======
>>>>>>> master
from utils.data_helper import data_specification, create_batch
from utils.cha_level_helper import output_sequence

parser = argparse.ArgumentParser()

<<<<<<< HEAD
# Training model
=======
# Training model
>>>>>>> master
parser.add_argument('--mode', type=str, default='train', help='Mode, test, train or validation')
parser.add_argument('--device', type=str, default='cpu', help='Training model with cpu or gpu')

# Training parameter
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--epochs', type=int, default=1000, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--display_step', type=int, default=200, help='Step of displaying accuracy')
parser.add_argument('--keep_prob', type=float, default=0.1, help='Probability of dropout')

# NN architecture Parameters
parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of rnn/lstm/gru cell')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in rnn/lstm/gru cell')
parser.add_argument('--rnn_celltype', type=str, default='gru', help='RNN cell type, rnn, gru, lstm')

# NN optimizer
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
parser.add_argument('--activation', type=str, default='tanh', help='Activation: sigmoid, tanh, relu')
parser.add_argument('--grad_clip', type=int, default=1, help='apply gradient clipping')
parser.add_argument('--layer_norm', type=bool, default=False, help='apply layer normalization')

# Training data Parameters
parser.add_argument('--num_features', type=int, default=39, help='Number of input feature')
parser.add_argument('--num_classes', type=int, default=30, help='Number of output classes')

# dir
dir_all = '/home/liangliang/Desktop/VUB_ThirdSemester/MasterThesis/practice/SpeechRecognitionProject1/LibriSpeechMFCCFeature/cha/dev-clean/0'
parser.add_argument('--data_dir', type=str, default=dir_all, help='Data directory')
parser.add_argument('--tensorboard_log_dir', type=str, default= './TENSORBOARD_LOG', help='TENSORBOARD_LOG directory')
parser.add_argument('--checkpoint_dir', type=str, default= './CHECKPOINT')

# get paser Argument
args = parser.parse_args()

################################
#     SESSION
################################
class SessionRun(object):
    def run_session(self, args):
        # get data

        ################################
        # get data and model handler
        ################################
        data, label = data_specification(args.mode, args.data_dir, 'npy')
        predict_file_name = 'predicted.csv'

        CHECKPOINT = args.checkpoint_dir
<<<<<<< HEAD


=======


>>>>>>> master
        max_sequence_length = 0
        for i in data:
            # print(i.shape)
            max_sequence_length = max(max_sequence_length, i.shape[1])
<<<<<<< HEAD

        batches = create_batch(args, data, max_sequence_length, label, args.mode)
        # model = brnn(args, max_sequence_length)
        model = brnnkeras(args, max_sequence_length)
        model.ctc_model.summary()
=======

        batches = create_batch(args, data, max_sequence_length, label, args.mode)
        model = brnn(args, max_sequence_length)

>>>>>>> master
        # if args.device_name == "gpu":
        #     self.device_name = "/gpu:0"
        # else:
        #     self.device_name = "/cpu:0"
<<<<<<< HEAD

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

=======

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

>>>>>>> master
        with tf.Session(graph=model.graph, config) as sess:
            if args.mode == "train":
                print('Initializing All the variables')
                sess.run(model.initializer)
            elif args.mode == "test":
                print('Restore from checkpoint')
                ck = tf.train.get_checkpoint_state(args.checkpoint_dir)
                if ck and ck.model_checkpoint_path:
                    model.saver.restore(sess, ck.model_checkpoint_path)
                print('Finished restoring')
<<<<<<< HEAD

=======

>>>>>>> master
            for epoch in range(args.epochs):
                start = time.time()
                cur_checkpoint_path = os.path.join(CHECKPOINT, '{:.0f}'.format(start))
                if not os.path.exists(cur_checkpoint_path):
                    os.makedirs(cur_checkpoint_path)

                if args.mode == 'train':
                    print('Epoch {} ================='.format(epoch + 1))
<<<<<<< HEAD

=======

>>>>>>> master

                errRate_list = np.zeros(len(batches))
                rand = np.random.permutation(len(batches))

                for rand_idx, batch_idx in enumerate(rand):
                    batch_data, (batch_label_indices, batch_label_value, batch_label_shape), batch_sequence_length = batches[batch_idx]
                    print("Batch_data get")
<<<<<<< HEAD
                    feeddict = {model.X: batch_data,
                                model.y_indices: batch_label_indices,
                                model.y_value: batch_label_value,
=======
                    feeddict = {model.X: batch_data,
                                model.y_indices: batch_label_indices,
                                model.y_value: batch_label_value,
>>>>>>> master
                                model.y_shape: batch_label_shape,
                                model.sequence_length: batch_sequence_length}

                    print("start training")
                    if args.mode == 'train':
<<<<<<< HEAD
                        _, batch_loss, batch_pred, batch_Y, batch_errRate = sess.run([model.optimizer,
                                                                                    model.loss,
=======
                        _, batch_loss, batch_pred, batch_Y, batch_errRate = sess.run([model.optimizer,
                                                                                    model.loss,
>>>>>>> master
                                                                                    model.pred,
                                                                                    model.Y,
                                                                                    model.errRate],
                                                                                    feed_dict = feeddict)
                        print("model train")
                        errRate_list[rand_idx] = batch_errRate
                        print('\n ,epoch:{},batch:{},train loss={:.3f},mean train Cha_er={:.3f}\n'.format(
                                    epoch+1, batch_idx, batch_loss, batch_errRate/args.batch_size))

                    elif args.mode == 'test':
<<<<<<< HEAD
                        batch_loss, batch_pred, batch_Y, batch_errRate = sess.run([model.loss,
=======
                        batch_loss, batch_pred, batch_Y, batch_errRate = sess.run([model.loss,
>>>>>>> master
                                                                                model.pred,
                                                                                model.Y,
                                                                                model.errRate],
                                                                                feed_dict = feeddict)
                        errRate_list[rand_idx] = batch_errRate
                        print('\n ,epoch:{},batch:{},train loss={:.3f},mean train Cha_er={:.3f}\n'.format(
                                    epoch+1, batch_idx, batch_loss, batch_errRate/args.batch_size))

                    elif args.mode == 'val':
<<<<<<< HEAD
                        batch_loss, batch_pred, batch_Y, batch_errRate = sess.run([model.optimizer,
                                                                                model.loss,
=======
                        batch_loss, batch_pred, batch_Y, batch_errRate = sess.run([model.optimizer,
                                                                                model.loss,
>>>>>>> master
                                                                                model.pred,
                                                                                model.Y,
                                                                                model.errRate],
                                                                                feed_dict = feeddict)
                        errRate_list[rand_idx] = batch_errRate
                        print('\n ,epoch:{},batch:{},train loss={:.3f},mean train Cha_er={:.3f}\n'.format(
                                    epoch+1, batch_idx, batch_loss, batch_errRate/args.batch_size))

                    if rand_idx % 1 == 0:
                        print('Ground Truth: ' + output_sequence(batch_Y))
                        print('Recognized As' + output_sequence(batch_pred))


                end = time.time()
                diff_time = end - start
                print('Epoch ' + str(epoch + 1) +'   '+ str(diff_time) + ' s')

                #save checkpoint
                if args.mode == 'train':
                    model.saver.save(sess, cur_checkpoint_path, global_step=epoch)

                if args.mode == 'test' or args.mode == 'val':
                    with open(predict_file_name, 'a') as f:
<<<<<<< HEAD
                            f.write('{},{},{},{}\n'.format(epoch,
=======
                            f.write('{},{},{},{}\n'.format(epoch,
>>>>>>> master
                                                        batch_idx,
                                                        output_sequence(batch_Y),
                                                        output_sequence(batch_pred)))


if __name__ == '__main__':
    sr = SessionRun()
<<<<<<< HEAD
    sr.run_session(args)
=======
    sr.run_session(args)
>>>>>>> master
