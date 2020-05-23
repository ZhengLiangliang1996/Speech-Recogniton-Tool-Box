"""
Author: Liangliang ZHENG
Time: 2020/3/06
Split the MFCC npy data in the samples files to training and test set

"""

import split_folders
import sys
import os
import random
import shutil

sys.path.append('..')


# Copy the file to train dev and test folder
def copy_file(filename_list, input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for f in filename_list:
        newPath = shutil.copy(input_path+f, output_path+f)
        # print(newPath)

def split_ratio(folder_name, train_ratio=0.8, test_ratio=0.1, dev_ratio=0.1):
    filenames = os.listdir(folder_name)
    filenames.sort()  # make sure that the filenames have a fixed order before shuffling
    random.seed(230)
    random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

    split_1 = int(train_ratio * len(filenames))
    split_2 = int((train_ratio + test_ratio) * len(filenames))

    train_filenames = filenames[:split_1]
    test_filenames = filenames[split_1:split_2]
    dev_filenames = filenames[split_2:]

    return train_filenames, test_filenames, dev_filenames

# Feature
feature_folder_path = '../samples/cha/stimuli3/0/feature/'
# Get splited file name
train_filenames, test_filenames, dev_filenames = split_ratio(feature_folder_path, 0.8, 0.2, 0.)
# copy file

# Train
train_feature_path = '../sampls/cha/stimuli3/0/feature_train/'
test_feature_path = '../samples/cha/stimuli3/0/feature_test/'
dev_feature_path = '../samples/cha/stimuli3/0/feature_dev/'

copy_file(train_filenames, feature_folder_path, train_feature_path)
copy_file(test_filenames, feature_folder_path, test_feature_path)
copy_file(dev_filenames, feature_folder_path, dev_feature_path)

# Label
label_folder_path = '../samples/cha/stimuli3/0/label/'
# Get splited file name
train_filenames_label, test_filenames_label, dev_filenames_label = split_ratio(label_folder_path, 0.8, 0.2, 0.)
# copy file

# Label
train_label_path = '../samples/cha/stimuli3/0/label_train/'
test_label_path = '../samples/cha/stimuli3/0/label_test/'
dev_label_path = '../samples/cha/stimuli3/0/label_dev/'

copy_file(train_filenames_label, label_folder_path, train_label_path)
copy_file(test_filenames_label, label_folder_path, test_label_path)
copy_file(dev_filenames_label, label_folder_path, dev_label_path)



