import glob
import os
import numpy as np

def data_specification(mode, dir, file_suffix):
    """ Get data according to mode and file_suffix

    Args:
        mode: mode of the dataset: train, test and dev
        dir:  directory of the dataset, only need to specify where train/test/dev file is located
        file_suffix: suffix of those feature file
    """
    train_data = []
    train_label= []

    for fname in glob.glob(os.path.join(dir, 'feature_'+mode, '*.'+file_suffix)):
        train_data.append(np.load(fname))

    for fname in glob.glob(os.path.join(dir, 'label_'+mode, '*.'+file_suffix)):
        train_label.append(np.load(fname))

    # total number of files
    path = dir + '/feature_'+ mode
    total = len(os.listdir(path))
    return train_data, train_label, total

# inspired by https://github.com/brianlan/automatic-speech-recognition/blob/master/src/brnn_ctc.py
def to_sparse_tensor(label, batch_idx):
    # definition of y
    # self.y_indices = tf.placeholder(tf.int64, shape = (None, 2)) # y_indices eg: ([2, 3], [1, 2])
    # self.y_value = tf.placeholder(tf.int64)
    # self.y_shape = tf.placeholder(tf.int64)

    # self.Y = tf.SparseTensor(self.y_indices, self.y_value, self.y_shape)
    indices = [] #index
    value = [] #value

    for i, idx in enumerate(batch_idx):
        for j, v in enumerate(label[idx]):
            indices.append([i, j])
            value.append(v)

    shape = [len(batch_idx), np.max(indices, axis=0)[1] + 1]
    return (np.array(indices), np.array(value), np.array(shape))


def create_batch(args, data, max_seq_length,label, mode = 'train'):

    num_samples = len(data)
    num_batches = num_samples // args.batch_size


    rand_idx = np.random.permutation(num_samples)

    final_batched = []

    for i in range(num_batches):
        start = args.batch_size * i
        end = min((i + 1) * args.batch_size, num_samples)
        batch_idx = rand_idx[start:end]
        batch_label = to_sparse_tensor(label, batch_idx)
        batch_data = np.zeros((max_seq_length, args.batch_size, args.num_features))
        sequence_length = np.zeros(args.batch_size)
        for batch, idx in enumerate(batch_idx):
            #self.X = tf.placeholder(tf.float32, shape = (max_seq_length, args.batch_size, args.num_features))
            #for Sparse Tensor
            # batch is ith of batch from rand_idx
            # idx is from 0~len(batch_idx)
            data_idx = data[idx]
            batch_data[0:data_idx.shape[1], batch, :] = np.reshape(data_idx, (data_idx.shape[1], args.num_features))
            sequence_length[batch] = data_idx.shape[1]
        final_batched.append((batch_data, batch_label, sequence_length))

    return final_batched
