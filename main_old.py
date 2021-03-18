# CSE 415
# Project Milestone B
# May 29, 2019
# Mourad Heddaya and Chris Pecunies

# Hidden Markov Model speech prediction application
# (Eventually) will allow for recording one-word user recodings to be
# interpreted to a word from a dictionary of words

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import matplotlib.pyplot as plt
import os
import pyaudio
import random
from sys import byteorder
from array import array
import wave
import matplotlib.colors as colors
from scipy.fftpack import dct

import forward as hmm

THRESHOLD = 99
CHUNK_SIZE = 512
FORMAT = pyaudio.paInt16
RATE = 44100

audio_directories = []
words = []
dir_to_word = dict()

def get_audio_info():
    words = []
    audio_directories= []
    audio_data = []
    for _, i in enumerate(os.listdir('audio')):
        for j in os.listdir('audio/' + i):
            audio_dir = 'audio/' + i + '/' + j
            data = wavfile.read(audio_dir)[1]
            audio_data.append(data)
            audio_directories.append(audio_dir)
            dir_to_word[audio_dir] = i
            if i not in words:
                words.append(i)
    return (words, audio_directories, audio_data)

#----- GET FEATURES USING PEAK FINDING ALGORITHM ------#
''' Uses STFT and peak finding to extract features from audio '''
def short_time_ft(audio_data):
    fs = 10e3
    stft_data = []
    for signal in audio_data:
        f, t, z = stft(signal, fs, nperseg=1000)
        stft_data.append((f, t, z))
    return stft_data

def single_stft(audio):
    fs = 10e3
    f, t, z = stft(audio, fs, nperseg=1000)
    return f, t, z

def get_audio_features(freq_data):
    peaks = []

    for stft in freq_data:
        f, t, z = stft
        for freq in f:
            for time in t:
                peaks.append(find_peaks(z))
    return peaks


# -----------------------------------------------------#
# ---- GET FEATURES USING MFCC ------------------------#
''' Uses MFCC as audio features from file '''

def get_mfcc(audio_path):
    sample_rate, signal = wavfile.read(audio_path)
    pre_emphasis = 0.97
    frame_size = 0.025
    frame_stride = 0.01
    NFFT = 512
    nfilt = 40
    # num_ceps = 12
    num_ceps = 6
    cep_lifter = 22
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc

def get_mfcc_for_each_directory():
    mfccs = dict.fromkeys(audio_directories)
    for directory in audio_directories:
        mfccs[directory] = get_mfcc(directory)
    return mfccs

# -----------------------------------------------------#
# add record from microphone here on in new module ->
    
def record(word = None, iteration = None):
    
    THRESHOLD = 100
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16
    RATE = 44100
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')
    is_silent = lambda audio: max(audio) < THRESHOLD
    
    def normalize(audio):
        MAXIMUM = 16384
        times = float(MAXIMUM)/max(abs(i) for i in audio)
    
        r = array('h')
        for i in audio:
            r.append(int(i*times))
        return r
    
    def trim(audio):
        def _trim(snd_data):
            snd_started = False
            r = array('h')
            for i in snd_data:
                if not snd_started and abs(i)>THRESHOLD:
                    snd_started = True
                    r.append(i)

                elif snd_started:
                    r.append(i)
            return r
        audio = _trim(audio)
        audio.reverse()
        audio = _trim(audio)
        audio.reverse()
        return audio
    
    print('Recording, speak now...')
    while 1:
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break
    print('Finished recording!')
    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()
    '''
    r = normalize(r)
    '''
    r = trim(r)
    if word is None:
        wf = wave.open('sample.wav', mode='wb')
    else:
        if iteration is None:
            wf = wave.open('audio/'+word+'/'+word+'_sample.wav')
        else:
            wf = wave.open('audio/'+word+'/'+word+iteration+'.wav')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(r)
    wf.close()
    return r

class HMM:

    def __init__(self, num_of_states):
        self.num_of_states = num_of_states
        self.rand_state = np.random.RandomState(1)
        self.pi = [random.uniform(0, 1) for x in range(3)]
        self.a = [[random.uniform(0,1) for x in range(3)] for y in range(3)]

    def step_forward(self, obs, state):
        # initilization
        trellis = [[0 for x in range(len(state))] for y in range(len(obs))]
        for s in range(len(state)):
            trellis[1][s] = self.pi[s] * b[s][w(1)]

        # recursion
        for t in range(2, len(obs)):
            for s in range(len(state)):
                trellis[s][t] = 0
                for sp in range(len(state)):
                    trellis[t][s] += trellis[t-1][sp] * self.a[sp][s]
                trellis[t][s] *= b[s][w(t)]

        # termination
        pw = 0
        for s in range(len(state)):
            pw += trellis[s][len(obs)]

        return pw

    def step_backwads():
        pass

from sklearn.model_selection import StratifiedShuffleSplit

def train_test():
    
    sss = StratifiedShuffleSplit(all_labels, test_size=0.1, random_state=0)
    for n,i in enumerate(all_obs):
        all_obs[n] /= all_obs[n].sum(axis=0)
        
    
    for train_index, test_index in sss:
        X_train, X_test = all_obs[train_index, ...], all_obs[test_index, ...]
        y_train, y_test = all_labels[train_index], all_labels[test_index]
    ys = set(all_labels)
    ms = [gmmhmm(7) for y in ys]
    
    _ = [model.train(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]
    ps1 = [model.test(X_test) for m in ms]
    res1 = np.vstack(ps1)
    predicted_label1 = np.argmax(res1, axis=0)
    dictionary = ['apple', 'banana', 'elephant', 'dog', 'frog', 'cat', 'jack', 'god', 'Intelligent', 'hello']
    spoken_word = []
    for i in predicted_label1:
        spoken_word.append(dictionary[i])
    print(spoken_word)
    missed = (predicted_label1 != y_test)
    print('Test accuracy: %.2f percent' % (100 * (1 - np.mean(missed))))


# if __name__ == '__main__':
#     words, audio_directories, audio_data = get_audio_info()
#     stft_data = short_time_ft(audio_data)
#     # audio = record()
#     # plt.plot(audio)
#     # f, t, Zxx = single_stft(audio)
#     # m = get_mfcc('sample.wav')
#     mfccs = get_mfcc_for_each_directory()
#     # plt.pcolormesh(t, f, np.abs(Zxx))
#     # plt.show()
#     # print(mfccs['audio/apple/apple07.wav'].T[0])
#     # print(mfccs.shape)
#     # for i in range(7):
#     #     hmm.generate_emissions(mfccs['audio/apple/apple07.wav'].T)
#     #     hmm.emissions_probs(mfccs['audio/apple/apple07.wav'].T)
#     #     hmm.forward_backward(mfccs['audio/apple/apple07.wav'].T)
#     all_obs = {'banana':np.array((15, 6, 32)), 'apple':np.array((15, 6, 32)), 'kiwi':np.array((15, 6, 32)), 'orange':np.array((15, 6, 32)), 'peach':np.array((15, 6, 32)), 'pineapple':np.array((15, 6, 32)),
#                'lime':np.array((15, 6, 32))}
#     # all_obs = {}
#     counts = [0,0,0,0,0,0,0]
#     for key, value in mfccs.items():
#         if 'banana' in key:
#             print(value.T[:,:32].shape)
#             print(all_obs['banana'][counts[0].shape])
#             all_obs['banana'][counts[0]] = value.T[:,:32]
#             counts[0] += 1
#         # elif 'apple' in key:
#         #     all_obs['apple'] = all_obs['apple'].append(value.T[:,:32])
#         # elif 'kiwi' in key:
#         #     all_obs['kiwi'] = all_obs['kiwi'].append(value.T[:,:32])
#         # elif 'orange' in key:
#         #     all_obs['orange'] = all_obs['orange'].append(value.T[:,:32])
#         # elif 'peach' in key:
#         #     all_obs['peach'] = all_obs['peach'].append(value.T[:,:32])
#         # elif 'pineapple' in key:
#         #     all_obs['pineapple'] = all_obs['pineapple'].append(value.T[:,:32])
#         # elif 'lime' in key:
#         #     all_obs['lime'] = all_obs['lime'].append(value.T[:,:32])
#
#     print(all_obs.shape)
#
#     # all_obs = np.zeros((105, 6, 32))
#     # # all_obs = {}
#     # for i in range(len(mfccs.values())):
#     #     all_obs[i] = list(mfccs.values())[i].T[:,:32]
#         # all_obs[]
#         # all_obs.append(value)
#     # print(all_obs.shape)
#     # temp = all_obs[1,:,:]
#     # temp2 = all_obs[0,:,:]
#     # all_words = []
#     # for word in words:
#     #     for i in range(15):
#     #         all_words.append(word)
#     # all_words = np.asarray(all_words)
#     #
#     # all_labels = []
#     # for i in range(7):
#     #     for t in range(15):
#     #         all_labels.append(i)
#     #
#     # # print('Labels and label indices', all_labels)
#     #
#     # from sklearn.model_selection import StratifiedShuffleSplit
#     #
#     # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
#     #
#     # for n, i in enumerate(all_obs):
#     #     all_obs[n] /= all_obs[n].sum(axis=0)
#     #
#     # for train_index, test_index in sss.split(all_obs, all_labels):
#     #     X_train, X_test = all_obs[train_index, ...], all_obs[test_index, ...]
#     #     y_train, y_test = all_words[train_index], all_words[test_index]
#     # print('Size of training matrix:', X_train.shape)
#     # print('Size of testing matrix:', X_test.shape)
#
#     # hmm.generate_emissions(X_train[1])
#     # # hmm.emissions_probs(X_train[0])
#     # # hmm.forward()
#     # # hmm.backward()
#     # hmm.forward_backward(X_train[1])
#     # res = hmm.transform(X_train[1])
#     # print(res)
#     #     hmm.forward_backward(mfccs['audio/apple/apple07.wav'].T)
#
#     # h = hmm.model(X_train[0])
#     # h.forward_backward(X_train[0])
#     #
#     # print(h.transform(X_test[0]))
#
#     # # ms = [gmmhmm(6) for y in ys]
#     # # ms = [m.fit(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]
#     # res = []
#     # for y in ys:
#     #     obs = X_train[y_train == y, :, :][0]
#     #     hmm.generate_emissions(obs)
#     #     hmm.forward_backward(obs)
#     #     res.append(hmm.transform(obs))
#     #
#     # hmm.model(X_train[0])
#
#     # ys = set(all_words)
#     # print(ys)
#     # ms = [hmm.model(X_train[y_train == y, :, :]) for y in ys]
#     # ms = [hmm.model(all_obs[y, :, :]) for y in range(0,15)]
#     # ys = set(all_words)
#     # ms = [hmm.model() for y in ys]
#     # _ = [m.train(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]
#     # for m,y in zip(ms, ys):
#     #    for index, t in enumerate(y_train):
#     #        if y == t:
#     #            m.train(X_train[index, :, :])
#     #
#     # print(ys)
#     # ps = [m.transform(X_test) for m in ms]
#     # # print(X_test)
#     # res = np.vstack(ps)
#     # predicted_labels = np.argmax(res, axis=0)
#     # # print(all_words)
#     # predictions=[]
#     # for p in predicted_labels:
#     #     predictions.append(list(ys)[p])
#     # print(predictions)
#     # print(y_test)
#     # missed = (predicted_labels != y_test)
#     # print('Test accuracy: %.2f percent' % (100 * (1 - np.mean(missed))))
#
#     # ps = [m.transform(X_test) for m in ms]
#
#     # print(test.transform(all_obs[22, :, :]))
#     # ps = [m.transform(all_obs[20, :, :]) for m in ms]
#     # res = np.vstack(ps)
#     # print(res)
#     # predicted_labels = np.argmax(res, axis=0)
#     # print(predicted_labels)
#     # print(all_words[predicted_labels*15])
#     # missed = (all_words[predicted_labels*15] != y_test)
#     # print(y_test)
#     # print('Test accuracy: %.2f percent' % (100 * (1 - np.mean(missed))))
#
#     # from sklearn.metrics import confusion_matrix
#     #
#     # cm = confusion_matrix(y_test, all_words[predicted_labels*15])
#     # plt.matshow(cm, cmap='gray')
#     # ax = plt.gca()
#     # _ = ax.set_xticklabels([" "] + [l[:2] for l in words])
#     # _ = ax.set_yticklabels([" "] + words)
#     # plt.title('Confusion matrix, single speaker')
#     # plt.ylabel('True label')
#     # plt.xlabel('Predicted label')
#     # plt.show()
#
#     # print(words)
#     # ms = [hmm.model() for w in words]
#     # for index, m in enumerate(ms):
#     #     m.train(all_obs[index*15:(index+1) * 15, :, :])
#     #
#     # for index, m in enumerate(ms):
#     #     print(m.transform(all_obs[95, :, :]))