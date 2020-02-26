"""
Author: Liangliang ZHENG
Time: 2020/2/25
Calculating MFCC feature from raw speech dataset

dataset used in this thesis: LibriSpeech
                             3 syllables dataset recorded from Bart de Boer

Inspired from: https://github.com/zzw922cn/Automatic_Speech_Recognition
"""


import os
import argparse
import glob
import sys
sys.path.append('/home/liangliang/Desktop/VUB_ThirdSemester/MasterThesis/SpeechRecognitiontoolbox')

import sklearn
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from python_speech_features import mfcc
from python_speech_features import delta


## original phonemes
phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

## cleaned phonemes
#phn = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']

def preprocess(root_directory):
    """
    Function to convert the stimuli3 file to .label file
    """
    for subdir, dirs, files in os.walk(root_directory):
        for f in files:
            # filename = os.path.join(subdir, f)
            if f.endswith(".wav"):
                label = f[:-6]
                subfile = os.path.join(root_directory, f[:-4]+".label")
                with open(subfile, 'w+') as sp:
                    sp.write(label)

def wav2feature(root_directory, save_directory, name, win_len, win_step, mode, feature_len, seq2seq, save):
    print(111)

    count = 0
    dirid = 0
    level = 'cha' if seq2seq is False else 'seq2seq'
    data_dir = os.path.join(root_directory, name)
    preprocess(data_dir)
    print(data_dir)
    print(222)

    for subdir, dirs, files in os.walk(data_dir):
        for f in files:
            fullFilename = os.path.join(subdir, f)
            filenameNoSuffix =  os.path.splitext(fullFilename)[0]
            if f.endswith('.wav'):
                rate = None
                sig = None    
                
                (rate,sig)= wav.read(fullFilename)
                print(rate)
                # Get MFCC feature
                mfcc_feat = mfcc(sig, rate, nfft=1103)
                delta_mfcc_feat = delta(mfcc_feat, 2)
                delta_delta = delta(delta_mfcc_feat, 2)
                #In each frame, coefficients are concatenated in (mfcc_feat,delta_mfcc_feat,delta_delta)
                feat = np.concatenate((mfcc_feat,delta_mfcc_feat,delta_delta),axis=1)
                # Normalize
                feat = preprocessing.scale(feat)
                feat = np.transpose(feat)
                print(feat.shape)

                labelFilename = filenameNoSuffix + '.label'
                with open(labelFilename,'r') as f:
                    characters = f.readline().strip().lower()
                targets = []
                if seq2seq is True:
                    targets.append(28)
                for c in characters:
                    if c == ' ':
                        targets.append(0)
                    elif c == "'":
                        targets.append(27)
                    else:
                        targets.append(ord(c)-96)
                if seq2seq is True:
                    targets.append(29)
                print(targets)
                if save:
                    count+=1
                    if count%4000 == 0:
                        dirid += 1
                    print('file index:',count)
                    print('dir index:',dirid)
                    label_dir = os.path.join(save_directory, level, name, str(dirid), 'label')
                    feat_dir = os.path.join(save_directory, level, name, str(dirid), 'feature')
                    if not os.path.isdir(label_dir):
                        os.makedirs(label_dir)
                    if not os.path.isdir(feat_dir):
                        os.makedirs(feat_dir)
                    featureFilename = os.path.join(feat_dir, filenameNoSuffix.split('/')[-1] +'.npy')
                    np.save(featureFilename,feat)
                    t_f = os.path.join(label_dir, filenameNoSuffix.split('/')[-1] +'.npy')
                    print(t_f)
                    np.save(t_f,targets)


if __name__ == '__main__':
    # character or phoneme
    parser = argparse.ArgumentParser(prog='data preprocess',
                                     description='Script to preprocess stimuli3 data')

    parser.add_argument("-path", help="stimuli3 dataset", 
                        type=str, default="/home/liangliang/Desktop/VUB_ThirdSemester/MasterThesis/SpeechRecognitiontoolbox/dataset")

    parser.add_argument("-save", help="Directory where preprocessed arrays are to be saved",
                        type=str, default="/home/liangliang/Desktop/VUB_ThirdSemester/MasterThesis/SpeechRecognitiontoolbox/samples")
    parser.add_argument("-n", "--name", help="Name of the dataset", type=str, default='stimuli3')

    parser.add_argument("-l", "--level", help="Level",
                        choices=['cha', 'phn'],
                        type=str, default='cha')
    parser.add_argument("-m", "--mode", help="Mode",
                        choices=['mfcc', 'fbank'],
                        type=str, default='mfcc')
    parser.add_argument("--featlen", help='Features length', type=int, default=13)
    parser.add_argument("-s", "--seq2seq", default=False,
                        help="set this flag to use seq2seq", action="store_true")

    parser.add_argument("-wl", "--winlen", type=float,
                        default=0.02, help="specify the window length of feature")

    parser.add_argument("-ws", "--winstep", type=float,
                        default=0.01, help="specify the window step length of feature")

if __name__ == "__main__":
    
    args = parser.parse_args()
    root_directory = args.path
    save_directory = args.save
    level = args.level
    mode = args.mode
    feature_len = args.featlen
    name = args.name
    seq2seq = args.seq2seq
    win_len = args.winlen
    win_step = args.winstep
    
    if root_directory == ".":
        root_directory = os.getcwd()
    if save_directory == ".":
        save_directory = os.getcwd()
    if not os.path.isdir(root_directory):
        raise ValueError("Root directory does not exist!")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    wav2feature(root_directory, save_directory, name=name, win_len=win_len, win_step=win_step,
                mode=mode, feature_len=feature_len, seq2seq=seq2seq, save=True)