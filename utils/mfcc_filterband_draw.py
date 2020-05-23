#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
File Description
"""

import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(12,4.5))
sr = 16000
mels= librosa.filters.mel(sr=sr, n_fft=512, n_mels=40,fmin=0, fmax=sr / 2)
mels /= np.max(mels, axis=-1)[:, None]
plt.plot(mels.T)
plt.grid(linestyle=':',linewidth=1)
plt.xlabel('Frequency')
plt.ylabel('Amplityde')

plt.show()


# MFCC spectrogram
y, sr = librosa.load('babibu_1.wav')
D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D, sr=sr,fmax=8000,n_mels=128)

plt.figure(figsize=(10, 4))
S_dB = librosa.power_to_db(S, ref=np.max)
display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram for speech signal babibu')
plt.tight_layout()
plt.show()
