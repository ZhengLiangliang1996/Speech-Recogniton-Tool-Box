"""
This file has not been used anymore, 

Instead, a ASR library has been used https://python-speech-features.readthedocs.io/en/latest/#python_speech_features.base.mfcc

Author: Liangliang ZHENG
Time: 2020/2/25
sCalculating MFCC feature from raw speech dataset


dataset used in this thesis: LibriSpeech
                             3 syllables dataset recorded from Bart de Boer

From: https://github.com/zzw922cn/Automatic_Speech_Recognition/blob/master/speechvalley/feature/core/calcmfcc.py
"""

import numpy 
import math 
from scipy.fftpack import dct

try:
    xrange(1)
except:
    xrange=range

def audio2frame(signal,frame_length,frame_step,winfunc=lambda x:numpy.ones((x,))):
    """ Framing audio signal. Uses numbers of samples as unit.
    Args:
    signal: 1-D numpy array.
	frame_length: In this situation, frame_length=samplerate*win_length, since we
        use numbers of samples as unit.
    frame_step:In this situation, frame_step=samplerate*win_step,
        representing the number of samples between the start point of adjacent frames.
	winfunc:lambda function, to generate a vector with shape (x,) filled with ones.
    Returns:
        frames*win: 2-D numpy array with shape (frames_num, frame_length).
    """
    signal_length=len(signal)
    # Use round() to ensure length and step are integer, considering that we use numbers
    # of samples as unit.
    frame_length=int(round(frame_length))
    frame_step=int(round(frame_step))
    if signal_length<=frame_length:
        frames_num=1
    else:
        frames_num=1+int(math.ceil((1.0*signal_length-frame_length)/frame_step))
    pad_length=int((frames_num-1)*frame_step+frame_length)
    # Padding zeros at the end of signal if pad_length > signal_length.
    zeros=numpy.zeros((pad_length-signal_length,))
    pad_signal=numpy.concatenate((signal,zeros))
    # Calculate the indice of signal for every sample in frames, shape (frams_nums, frams_length)
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(
        numpy.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T
    indices=numpy.array(indices,dtype=numpy.int32)
    # Get signal data according to indices.
    frames=pad_signal[indices]
    win=numpy.tile(winfunc(frame_length),(frames_num,1))
    return frames*win

def spectrum_magnitude(frames,NFFT):
    '''Apply FFT and Calculate magnitude of the spectrum.
    Args:
        frames: 2-D frames array calculated by audio2frame(...).
        NFFT:FFT size.
    Returns:
        Return magnitude of the spectrum after FFT, with shape (frames_num, NFFT).
    '''
    complex_spectrum=numpy.fft.rfft(frames,NFFT)
    return numpy.absolute(complex_spectrum)

def spectrum_power(frames,NFFT):
    """Calculate power spectrum for every frame after FFT.
    Args:
        frames: 2-D frames array calculated by audio2frame(...).
        NFFT:FFT size
    Returns:
        Power spectrum: PS = magnitude^2/NFFT
    """
    return 1.0/NFFT * numpy.square(spectrum_magnitude(frames,NFFT))

def log_spectrum_power(frames,NFFT,norm=1):
    '''Calculate log power spectrum.
    Args:
        frames:2-D frames array calculated by audio2frame(...)
        NFFT：FFT size
        norm: Norm.
    '''
    spec_power=spectrum_power(frames,NFFT)
    # In case of calculating log0, we set 0 in spec_power to 0.
    spec_power[spec_power<1e-30]=1e-30
    log_spec_power=10*numpy.log10(spec_power)
    if norm:
        return log_spec_power-numpy.max(log_spec_power)
    else:
        return log_spec_power

def pre_emphasis(signal,coefficient=0.95):
    '''Pre-emphasis.
    Args:
        signal: 1-D numpy array.
        coefficient:Coefficient for pre-emphasis. Defauted to 0.95.
    Returns:
        pre-emphasis signal.
    '''
    return numpy.append(signal[0],signal[1:]-coefficient*signal[:-1])



def hz2mel(hz):
    """Convert frequency to Mel frequency.
    Args:
        hz: Frequency.
    Returns:
        Mel frequency.
    """
    return 2595*numpy.log10(1+hz/700.0)

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
    mel_points=numpy.linspace(low_mel,high_mel,filters_num+2)
    # Convert Mel frequency to frequency and find corresponding position.
    hz_points=mel2hz(mel_points)
    # Find corresponding position of these hz_points in fft.
    bin=numpy.floor((NFFT+1)*hz_points/samplerate)
    # Build Mel filters' expression.First and third points of each filter are zeros.
    fbank=numpy.zeros([filters_num,NFFT//2+1])
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
        nframes,ncoeff=numpy.shape(cepstra)
        n=numpy.arange(ncoeff)
        lift=1+(L/2)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        return cepstra

def calcMFCC(signal,samplerate=16000,win_length=0.025,win_step=0.01,feature_len=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True,mode='mfcc'):
    """Caculate Features.
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
        2-D numpy array with shape (NUMFRAMES, features). Each frame containing feature_len of features.
    """
    filters_num = 2*feature_len
    feat,energy=fbank(signal,samplerate,win_length,win_step,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff)
    feat=numpy.log(feat)
    # Performing DCT and get first 13 coefficients
    if mode == 'mfcc':
        feat=dct(feat,type=2,axis=1,norm='ortho')[:,:feature_len]
        feat=lifter(feat,cep_lifter)
    elif mode == 'fbank':
        feat = feat[:,:feature_len]
    if appendEnergy:
        # Replace the first coefficient with logE and get 2-13 coefficients.
        feat[:,0]=numpy.log(energy)
    return feat

def fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    """Perform pre-emphasis -> framing -> get magnitude -> FFT -> Mel Filtering.
    Args:
        signal: 1-D numpy array.
        samplerate: Sampling rate. Defaulted to 16KHz.
        win_length: Window length. Defaulted to 0.025, which is 25ms/frame.
        win_step: Interval between the start points of adjacent frames.
            Defaulted to 0.01, which is 10ms.
        cep_num: Numbers of cepstral coefficients. Defaulted to 13.
        filters_num: Numbers of filters. Defaulted to 26.
        NFFT: Size of FFT. Defaulted to 512.
        low_freq: Lowest frequency.
        high_freq: Highest frequency.
        pre_emphasis_coeff: Coefficient for pre-emphasis. Pre-emphasis increase
            the energy of signal at higher frequency. Defaulted to 0.97.
    Returns:
        feat: Features.
        energy: Energy.
    """
    # Calculate the highest frequency.
    high_freq=high_freq or samplerate/2
    # Pre-emphasis
    signal=pre_emphasis(signal,pre_emphasis_coeff)
    # rames: 2-D numpy array with shape (frame_num, frame_length)
    frames=audio2frame(signal,win_length*samplerate,win_step*samplerate)
    # Caculate energy and modify all zeros to eps.
    spec_power=spectrum_power(frames,NFFT)
    energy=numpy.sum(spec_power,1)
    energy=numpy.where(energy==0,numpy.finfo(float).eps,energy)
    # Get Mel filter banks.
    fb=get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq)
    # Get MFCC and modify all zeros to eps.
    feat=numpy.dot(spec_power,fb.T)
    feat=numpy.where(feat==0,numpy.finfo(float).eps,feat)

    return feat,energy

def calcfeat_delta_delta(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True,mode='mfcc',feature_len=13):
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
    print("1111")
    filters_num = 2*feature_len
    feat = calcMFCC(signal,samplerate,win_length,win_step,feature_len,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff,cep_lifter,appendEnergy,mode=mode)   #首先获取13个一般MFCC系数
    feat_delta = delta(feat)
    feat_delta_delta = delta(feat_delta)

    result = numpy.concatenate((feat,feat_delta,feat_delta_delta),axis=1)
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
    feat = numpy.concatenate(([feat[0] for i in range(N)], feat, [feat[-1] for i in range(N)]))
    denom = sum([2*i*i for i in range(1,N+1)])
    dfeat = []
    for j in range(NUMFRAMES):
        dfeat.append(numpy.sum([n*feat[N+j+n] for n in range(-1*N,N+1)], axis=0)/denom)
    return dfeat