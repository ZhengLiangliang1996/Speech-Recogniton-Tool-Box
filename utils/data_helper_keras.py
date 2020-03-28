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

def create_batch(args, data, max_seq_length, label):
    while True:
        data_length = len(data)
        num_batches = data_length // args.batch_size

        # final batched
        # shuffle data with random index
        rand_idx = np.random.permutation(data_length)
        print(np.shape(label))
        for i in range(num_batches):
            # starting batch index
            start = args.batch_size * i
            # end batch index
            end = min((i + 1) * args.batch_size, data_length)
            # assign batch index to this batch
            batch_idx = rand_idx[start:end]
            # initialize the array
            batch_data = np.zeros([args.batch_size, max_seq_length, args.num_features])
            batch_label = np.ones([args.batch_size, max_seq_length])

            # input_length
            input_length = np.zeros([args.batch_size, 1])
            label_length = np.zeros([args.batch_size, 1])


            for num_batch, index in enumerate(batch_idx):
                # batch data
                batched_data = data[index]
                batched_data = np.transpose(batched_data)
                # in my case, the input length is in the second shape 1
                input_length[num_batch] = batched_data.shape[0]
                # same here
                batch_data[num_batch, 0:batched_data.shape[0],:] = batched_data

                # batch label
                batched_label = np.array(label[index])
                batched_label_len = len(batched_label)
                batch_label[num_batch, 0:batched_label_len] = batched_label
                label_length[num_batch] = batched_label_len

            #print(batch_label)
            #save it to the ctc dictionary
            outputs = {'ctc': np.zeros([args.batch_size])}

            inputs = {'the_input': batch_data,
                      'the_labels': batch_label,
                      'input_length': input_length,
                      'label_length': label_length}

            yield inputs, outputs

