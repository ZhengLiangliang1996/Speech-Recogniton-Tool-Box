"""
Author: Liangliang ZHENG
Time: 2020/2/25
Calculating MFCC feature from raw speech dataset

dataset used in this thesis: LibriSpeech
                             3 syllables dataset recorded from Bart de Boer

Inspired from: https://github.com/zzw922cn/Automatic_Speech_Recognition/blob/master/speechvalley/feature/core/calcmfcc.py
"""

import numpy as np
import math 
from scipy.fftpack import dct

try:
    xrange(1)
except:
    xrange=range

def pre_emphasis(signal, coef = 0.95):
    """ 
    Args: 
        signal: 1D numpy array 
        coef: coefficient for pre-emphasis
    
    Return:
        processed signal
    """
    return np.append(signal[0], signal[1:] - coef * signal[:-1]) 

def windowing(signal, frame_length, frame_step, winfunc=lambda x: np.one((x, ))):
    """
    Args:
        signal: 1D numpy array
        frame_length: frame_length = samplerate*win_length, number of samples as unit
        frame_step: frame_length = samplerate*win_step
        winfunc: lambda function, genearte a vector with shape(x, ) filled with ones

    Return:
        frames*win: 2D numpy array with shape(frames_num, frame_length)
    """
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    if signal_length <= frame_length:
        frames_num = 1
    else:
        frames_num = 1 + int(math.ceil((1.0*signal_length-frame_length)/frame_step))

    pad_length=int((frames_num-1)*frame_step+frame_length)
    # Padding zeros at the end of signal if pad_length > signal_length.
    zeros=np.zeros((pad_length-signal_length,))
    pad_signal=np.concatenate((signal,zeros))
    # Calculate the indice of signal for every sample in frames, shape (frams_nums, frams_length)
    indices=np.tile(np.arange(0,frame_length),(frames_num,1))+np.tile(
        np.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T
    indices=np.array(indices,dtype=np.int32)
    # Get signal data according to indices.
    frames=pad_signal[indices]
    win=np.tile(winfunc(frame_length),(frames_num,1))
    return frames*win

def spectrum_power(frames, nfft):
    """ 
    Args: 
        frames: 2D frames array calculated by windowing
        nfft: fft size
    
    Return:
        Power spectrum: magnitude ^ 2 / nfft
    """    

    complex_spectrum = np.fft.rfft(frames. nfft)
    return 1.0/nfft * np.square(np.absolute(complex_spectrum))

def hz2mel(hz):
    """Convert frequency to Mel frequency.
    Args:
        hz: Frequency.
    Returns:
        Mel frequency.
    """
    return 2595*np.log10(1+hz/700.0)

def mel2hz(mel):
    """Convert Mel frequency to frequency.
    Args:
        mel:Mel frequency
    Returns:
        Frequency.
    """
    return 700*(10**(mel/2595.0)-1)


def get_filter_banks(filters_num=20,NFFT=512,samplerate=16000,low_freq=0,high_freq=None):
    '''Calculate Mel filter banks.
    Args:
        filters_num: Numbers of Mel filters.
        NFFT:FFT size. Defaulted to 512.
        samplerate: Sampling rate. Defaulted to 16KHz.
        low_freq: Lowest frequency.
        high_freq: Highest frequency.
    '''
    # Convert frequency to Mel frequency.
    low_mel=hz2mel(low_freq)
    high_mel=hz2mel(high_freq)
    # Insert filters_num of points between low_mel and high_mel. In total there are filters_num+2 points
    mel_points=np.linspace(low_mel,high_mel,filters_num+2)
    # Convert Mel frequency to frequency and find corresponding position.
    hz_points=mel2hz(mel_points)
    # Find corresponding position of these hz_points in fft.
    bin=np.floor((NFFT+1)*hz_points/samplerate)
    # Build Mel filters' expression.First and third points of each filter are zeros.
    fbank=np.zeros([filters_num,NFFT/2+1])
    for j in xrange(0,filters_num):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i]=(i-bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i]=(bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra,L=22):
    '''Lifter function.
    Args:
        cepstra: MFCC coefficients.
        L: Numbers of lifters. Defaulted to 22.
    '''
    if L>0:
        nframes,ncoeff=np.shape(cepstra)
        n=np.arange(ncoeff)
        lift=1+(L/2)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        return cepstra

def cal_mfcc(original_signal, sample_rate = 16000, win_length = 0.025, win_step = 0.01, feature_len = 13, filters_num = 26, nfft = 512, freq_min = 0, freq_high = None, pre_emphasis_coef = 0.97, cep_lifter = 22, appendEnergy=True, mode='mfcc'):
    """
    Args:
        original_signal: 1D numpy array 
        sample_rate: sample rate
        window_length: 25ms/frame
        window_step: start points of adjacent frame 0.01(10ms)
        feature_len: numbers of features, default to 13
        filters_num: number of filters, default to 26
        n_fft: size of fft, default to 512
        freq_min
        fre_high
        pre_emphasis_coeff: pre-emphasis increase the energy signal at higher frequency
        cep_lifter: 
        appendEnergy: whether to append energy.
        mode: mfcc

    Returns:
        2-D numpy array with shape(num_frames, features) 
        Each frame containing feature_len of features.
    """
    filter_num = 2 * feature_len
    fre_high = freq_high or sample_rate/2
    # pre-emphasis
    signal = pre_emphasis(signal, pre_emphasis_coeff)
    # windowing 
    frames = windowing(signal, win_length * sample_rate, win_step * sample_rate)
    # Energy 
    spec_power = spectrum_power(frames, nfft)
    energy = np.sum(spec_power, 1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    # get Mel filter filter bank
    fb=get_filter_banks(filters_num,nfft,sample_rate,freq_min,freq_high)
    # Get MFCC and modify all zeros to eps.
    feat=np.dot(spec_power,fb.T)
    feat=np.where(feat==0,np.finfo(float).eps,feat)
    feat=np.log(feat)
    feat=dct(feat,type=2,axis=1,norm='ortho')[:,:feature_len]
    feat=lifter(feat,cep_lifter)

def calcfeat_delta_delta(signal,samplerate=16000,win_length=0.025,win_step=0.01, feature_len=13, filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True,mode='mfcc'):
    """Calculate features, fist order difference, and second order difference coefficients.
        13 Mel-Frequency Cepstral Coefficients(MFCC), 13 first order difference
       coefficients, and 13 second order difference coefficients. There are 39 features
       in total.
    Args:
        signal: 1-D numpy array.
        samplerate: Sampling rate. Defaulted to 16KHz.
        win_length: Window length. Defaulted to 0.025, which is 25ms/frame.
        win_step: Interval between the start points of adjacent frames.
            Defaulted to 0.01, which is 10ms.
        feature_len: Numbers of features. Defaulted to 13.
        filters_num: Numbers of filters. Defaulted to 26.
        NFFT: Size of FFT. Defaulted to 512.
        low_freq: Lowest frequency.
        high_freq: Highest frequency.
        pre_emphasis_coeff: Coefficient for pre-emphasis. Pre-emphasis increase
            the energy of signal at higher frequency. Defaulted to 0.97.
        cep_lifter: Numbers of lifter for cepstral. Defaulted to 22.
        appendEnergy: Wheter to append energy. Defaulted to True.
        mode: 'mfcc' or 'fbank'.
            'mfcc': Mel-Frequency Cepstral Coefficients(MFCC).
                    Complete process: Mel filtering -> log -> DCT.
            'fbank': Apply Mel filtering -> log.
    Returns:
        2-D numpy array with shape:(NUMFRAMES, 39). In each frame, coefficients are
            concatenated in (feature, delta features, delta delta feature) way.
    """
    filters_num = 2*feature_len
    feat = cal_mfcc(signal,samplerate,win_length,win_step,feature_len,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff,cep_lifter,appendEnergy,mode=mode)   #首先获取13个一般MFCC系数
    feat_delta = delta(feat)
    feat_delta_delta = delta(feat_delta)

    result = np.concatenate((feat,feat_delta,feat_delta_delta),axis=1)
    return result

def delta(feat, N=2):
    """Compute delta features from a feature vector sequence.
    Args:
        feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
        N: For each frame, calculate delta features based on preceding and following N frames.
    Returns:
        A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    NUMFRAMES = len(feat)
    feat = np.concatenate(([feat[0] for i in range(N)], feat, [feat[-1] for i in range(N)]))
    denom = sum([2*i*i for i in range(1,N+1)])
    dfeat = []
    for j in range(NUMFRAMES):
        dfeat.append(np.sum([n*feat[N+j+n] for n in range(-1*N,N+1)], axis=0)/denom)
    return dfeat