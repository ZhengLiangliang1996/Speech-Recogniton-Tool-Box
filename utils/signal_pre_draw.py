#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
File Description
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import numpy.fft as nf
import matplotlib.pyplot as plt
import python_speech_features as sf
# load the speech
sample_rate, sigs  = wavfile.read('babibu_1.wav')

print(sample_rate)
print(sigs.shape)

# draw the sinal with x time and amplitude
# sigs = sigs / (2 ** 15)
times = np.arange(len(sigs)) / sample_rate
freqs = nf.fftfreq(sigs.size, 1 / sample_rate)
ffts = nf.fft(sigs)
pows = np.abs(ffts)
plt.figure('Audio',figsize=(12,5))
plt.subplot(121)
plt.title('Time Domain')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Signal', fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')
plt.plot(times, sigs, c='dodgerblue', label='babibu.wav')
plt.legend()
plt.subplot(122)
plt.title('Frequency Domain')
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')
plt.plot(freqs[freqs >= 0], pows[freqs >= 0], c='orangered', label='Power')
plt.legend()
plt.tight_layout()
plt.show()

# after pre-emphasis
pre_emphasis = 0.97
emphasized_signal = np.append(sigs[1], sigs[1:] - pre_emphasis * sigs[:-1])
plt.figure('Emphasis')
plt.title('Speech After pre-emphasis')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Signal', fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')
plt.plot(times, emphasized_signal, c='dodgerblue', label='pre_emphasis=0.97')
plt.show()

