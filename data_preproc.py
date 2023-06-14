#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 22:28:49 2023

@author: avk256
"""
import dask.dataframe as dd
import dask.array as da
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM, Reshape, Input, Lambda, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import line_profiler
import math
from tqdm import tqdm
import pdb
from sklearn.model_selection import StratifiedShuffleSplit
import re
import time
import random
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras import optimizers, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.models import Sequential
from tensorflow import keras
import tensorflow_io as tfio
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import wave
import librosa.display
import soundfile as sf
import librosa
import plotly.express as px
import seaborn as sns
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools

import h5py
import sys
import pickle as cPickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# import librosa for analysing audio signals : visualize audio, display the spectogram

# import librosa for analysing audio signals : visualize audio, display the spectogram


# import wav for reading and writing wav files

# import IPython.dispaly for playing audio in Jupter notebook
# import IPython.display as ipd


profile = line_profiler.LineProfiler()

tf.autograph.set_verbosity(3)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.set_printoptions(threshold=sys.maxsize)
###############################################################################


def data_splitting(X, y):
    """
    Split datasets.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    X_val : TYPE
        DESCRIPTION.
    y_val : TYPE
        DESCRIPTION.

    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
    # sss.get_n_splits(table_dur_df['file name'], table_dur_df['queen status'])

    for i, (train_index, test_index) in enumerate(sss.split(X, y)):

        #     print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    ratio = int(0.3*len(X_train))
    X_val, y_val = X_train[:ratio], y_train[:ratio]
    X_train.drop(index=X_train.index[:ratio], axis=0, inplace=True)
    y_train.drop(index=y_train.index[:ratio], axis=0, inplace=True)

    return X_train, y_train, X_test, y_test, X_val, y_val

# def find_sound(self, filename):
#     return ["{} {}".format(index1,index2) for index1,value1 in enumerate(self.sounds_status) for index2,value2 in enumerate(value1) for index3,value3 in enumerate(value2)  if value3.split('/')[-1][:-14]==filename[:-4]]

# def wav_join(self, cat, sound):
# #         print('wav_join')
# #         pdb.set_trace()
#     x, sr = librosa.load(self.sounds_status[cat][sound][0])
# #         print("File " + self.sounds_status[cat][sound][0] + " loaded")
#     filename = (self.sounds_status[cat][sound][0].split('/')[-1]).split('__')[0] + '.wav'

#     for segment in range(1, len(self.sounds_status[cat][sound])):
#         y, sr = librosa.load(self.sounds_status[cat][sound][segment])
# #             print("File " + self.sounds_status[cat][sound][segment] + " loaded")
#         x = np.append(x,y)
# #     sf.write(filename, x, sr, subtype='PCM_24')
#     return x, sr

# def sound_load(self, file_name):
# #         print('sound_load')
# #         pdb.set_trace()
#     indexes = self.find_sound(file_name)
#     cat, sound = indexes[0].split(' ')
#     cat, sound = int(cat), int(sound)
#     x, sr = self.wav_join(cat, sound)

#     return x, sr


def get_sound_list(path='./data/data/'):
    sound_list = []
    for dirname, _, filenames in os.walk('./data/data/'):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            sound_list.append(os.path.join(dirname, filename))
    return sound_list


# data = "./data/data/sound_files/"
# wav_files = []
# for path, subdirs, files in os.walk(data):
#     for name in files:
#         wav_files.append(os.path.join(path, name))

# print(table_df.columns)


def define_status(df, queen_status_dict, sound_list):

    no_files = []
    sounds_status = [0]*len(queen_status_dict)
    files_status = [0]*len(queen_status_dict)
    for i in range(0, len(queen_status_dict)):
        files_status[i] = list(
            table_df['file name'].loc[table_df['queen status'] == i])

    # print(files_status[3])

    for i in range(0, 4):
        sounds = []
        for j in range(len(files_status[i])):

            r = re.compile(".*" + files_status[i][j][:-4])
            newlist = list(filter(r.match, sound_list))  # Read Note below
            newlist.sort()
            if len(newlist) == 0:
                no_files.append(files_status[i][j])
                continue
    #             print(i, j)
    #             print(files_status[i][j])
    #             print(newlist)

            sounds.append(newlist)
        sounds_status[i] = sounds

    return sounds_status


def define_dur(sounds_status):

    filenames = []
    filedurat = []

    for cat in range(0, 4):
        for sound in range(len(sounds_status[cat])):
            filenames.append((sounds_status[cat][sound][0].split(
                '/')[-1]).split('__')[0] + '.raw')
            filedurat.append(len(sounds_status[cat][sound]))
    # sound_durat = dict(zip(filenames, filedurat))
    sound_durat = {'file name': filenames, 'durat': filedurat}
    sound_durat_df = pd.DataFrame.from_dict(sound_durat)

    table_dur_df = pd.merge(table_df, sound_durat_df, on='file name')

    return table_dur_df


def data_filter_dur(table_dur_df, dur):

    table_dur_df = table_dur_df[table_dur_df['durat'] == dur]
    table_dur_df = table_dur_df.reset_index()

    return table_dur_df


def samples_permut(table_dur_df, n_samples):

    for i in range(len(table_dur_df)):
        permut = list(itertools.permutations(
            range(table_dur_df['durat'][i]), n_samples))
        sample_permut = np.random.permutation(permut)[0:10]
        table_dur_df.at[i, 'samples'] = " ".join(list(map(str, sample_permut)))

    return table_dur_df


def manipulate(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def find_sound(lst, filename):
    return ["{} {}".format(index1, index2) for index1, value1 in enumerate(lst) for index2, value2 in enumerate(value1) for index3, value3 in enumerate(value2) if value3.split('/')[-1][:-14] == filename[:-4]]


def wav_join(sounds_status, cat, sound):
    x, sr = librosa.load(sounds_status[cat][sound][0])
    # print("File " + sounds_status[cat][sound][0] + " loaded")
    filename = (sounds_status[cat][sound][0].split(
        '/')[-1]).split('__')[0] + '.wav'

    for segment in range(1, len(sounds_status[cat][sound])):
        y, sr = librosa.load(sounds_status[cat][sound][segment])
        # print("File " + sounds_status[cat][sound][segment] + " loaded")
        x = np.append(x, y)
#     sf.write(filename, x, sr, subtype='PCM_24')
    return x, sr


def sound_load(file_name, sounds_status):

    indexes = find_sound(sounds_status, file_name)
    cat, sound = indexes[0].split(' ')
    cat, sound = int(cat), int(sound)
    x, sr = wav_join(sounds_status, cat, sound)

    return x, sr


def extract_audio_clip(df_table, sounds_status, index, model1, model2):
    #         print('extract_audio_clip')
    #         pdb.set_trace()
    record_name = df_table.loc[index]["file name"]  # train_clip_folder
    signal, srate = sound_load(record_name, sounds_status)

    # return signal, extract_featur(model1, model2, signal), extract_featur(model1, model2, signal,"VGGISH"), srate
    return signal, extract_featur(model1, model2, signal), srate


def extract_featur(prenet_model1, prenet_model2, signal, model="YAMNET"):
    #         print('extract_featur')
    #         pdb.set_trace()

    if model == "VGGISH":
        res = prenet_model2(signal)[1]

    elif model == "YAMNET":
        # return prenet_model1(signal)
        res = np.array(prenet_model2(signal)[1])
        # print(res.shape)
    return res


def spec_transp_scal(signal, sr, **stft_param):
    #         print('spec_transp_scal')
    #         pdb.set_trace()

    # librosa.amplitude_to_db(abs(librosa.stft(signal, **self.stft_param)))
    signal = librosa.feature.melspectrogram(y=signal, sr=sr, **stft_param)
    # transpose because we going to reshape inside the model and use a LSTM to extract time features
    signal = np.transpose(signal[-int(len(signal)/4):])
    return ((signal - 1.0)/90.0)  # min max scale


def manipulate(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def noise_signal(signal):

    #         print('noise_signal')
    #         pdb.set_trace()
    noise_factor = random.uniform(0.0001, 0.001)
    signal = manipulate(signal, noise_factor)

    return signal


def model_Dense(input_shape, n_clases=4, saved_file=None):
    initializer = tf.random_normal_initializer(0, 0.03)
    input_layer = Input(input_shape)

    y = Flatten()(input_layer)
    y = Dense(128, kernel_initializer=initializer)(y)

    y = Dense(128, kernel_initializer=initializer)(
        y)  # input_shape=features.shape[1:]
    y = Activation('relu')(y)
    y = BatchNormalization()(y)
    # model.add(Dropout(0.25))

    # y = Dense(128, kernel_initializer = initializer)(y)#input_shape=features.shape[1:]
    #y = Activation('relu')(y)
    #y = BatchNormalization()(y)
    # model.add(Dropout(0.25))
    # y = Dense(128, kernel_initializer = initializer)(y)#input_shape=features.shape[1:]
    #y = Activation('relu')(y)
    #y = BatchNormalization()(y)

    y = Dense(n_clases, kernel_initializer=initializer)(y)
    out = Activation('softmax')(y)
    #sgd = optimizers.SGD(lr=0.1, decay=1e-3, momentum=1e-3)
    model = Model(inputs=[input_layer], outputs=out,)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])
    #tf.keras.utils.plot_model(model, to_file='NN_model.jpg', show_shapes=True)
    if (saved_file):
        try:
            # model.load_model(saved_file)
            model.load_weights(saved_file)
            print("Pesos cargados")
        except:
            print("No se puede cargar los pesos")

    model.summary()
    return model


def cat_weights(df, target):

    cat_count = df[target].value_counts()
    cat_count = cat_count.to_dict()
    max_cat = max(cat_count.values())
    cat_count1 = {k: math.ceil((v / max_cat)**(-1))-1 for max_cat in (
        max(cat_count.values()),) for k, v in cat_count.items()}

    return cat_count1


###############################################################################

# prenet_model1 = hub.load('https://tfhub.dev/google/vggish/1')
prenet_model2 = hub.load('https://tfhub.dev/google/yamnet/1')
prenet_model1 = prenet_model2

table_df = pd.read_csv('./data/data/all_data_updated.csv')
queen_status_dict = {0: 'original queen', 1: 'not present',
                     2: 'present and rejected', 3: 'present and newly accepted'}

sound_list = get_sound_list()
sounds_status = define_status(table_df, queen_status_dict, sound_list)
table_dur_df = define_dur(sounds_status)
table_dur_df = data_filter_dur(table_dur_df, 6)
table_dur_df = samples_permut(table_dur_df, 3)

X_train, y_train, X_test, y_test, X_val, y_val = data_splitting(
    table_dur_df['file name'], table_dur_df['queen status'])
print(len(X_train))
print(len(X_test))
print(len(X_val))

df_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
df_train = df_train.reset_index(drop=True)
print(df_train)

df_test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
df_test = df_test.reset_index(drop=True)
print(df_test)

df_val = pd.concat([pd.DataFrame(X_val), pd.DataFrame(y_val)], axis=1)
df_val = df_val.reset_index(drop=True)
print(df_val)

stft_param = {"n_fft": 1100, "hop_length": 512}
# test

# test = df_val
# test = test.reset_index(drop=True)

# print(len(test))
# aug_weights = cat_weights(test, 'queen status')

# stft_param = {"n_fft": 1100, "hop_length": 512}

# signal, signal_ym, sr = extract_audio_clip(
#     test, sounds_status, 3, prenet_model1, prenet_model2)
# # signal_ym = signal_ym.numpy()
# # signal_vgg = signal_vgg.numpy()

# signal_spec = spec_transp_scal(signal, sr, **stft_param)

# sf.write('test.wav', signal, sr, subtype='PCM_24')

# signal_aug = noise_signal(signal)
# sf.write('test_aug.wav', signal_aug, sr, subtype='PCM_24')


def data_gen(df, sounds_status, prenet_model1, prenet_model2,  filename, mode,
             **stft_param):
    """
    Generate dataset and save to file.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    sounds_status : TYPE
        DESCRIPTION.
    prenet_model1 : TYPE
        DESCRIPTION.
    prenet_model2 : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.
    **stft_param : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    spec_list = []
    ym_list = []
    # vgg_list = []
    y_list = []
    for i in tqdm(range(len(df))):
        signal, signal_ym, sr = extract_audio_clip(
            df, sounds_status, i, prenet_model1, prenet_model2)

        ym_list.append(signal_ym)
        # vgg_list.append(signal_vgg)
        spec_list.append(spec_transp_scal(signal, sr, **stft_param))
        aug_weights = cat_weights(df, 'queen status')
        y_list.append(df['queen status'][i])
        for j in range(aug_weights[df['queen status'][i]]):
            signal_aug = noise_signal(signal)
            ym_list.append(extract_featur(
                prenet_model1, prenet_model2, signal_aug))
            # vgg_list.append(extract_featur(prenet_model1, prenet_model2, signal_aug,"VGGISH"))
            spec_list.append(spec_transp_scal(signal_aug, sr, **stft_param))
            y_list.append(df['queen status'][i])
    df_res = pd.DataFrame(list(zip(ym_list, spec_list, y_list)),
                          columns=['ym', 'spec', 'y'])

    print('\n Data file creating...')

    # h5File = filename
    # df_res.to_hdf(h5File, "/data/test_data")
    # type(df_res['ym'][0])

    # df_res.to_csv(filename, float_format='%.3f')

    df_res.to_pickle(filename + '_data.pkl')

    # f = open(filename + '_data.pkl', 'wb')
    # pickler = cPickle.Pickler(f)
    # if mode == 'ym':
    #     for e in ym_list:
    #         pickler.dump(e)
    # if mode == 'spec':
    #     for e in spec_list:
    #         pickler.dump(e)
    # if mode == 'y':
    #     for e in y_list:
    #         pickler.dump(e)
    # f.close()

    print('Data file is created')
    return 0


data_gen(df_train, sounds_status, prenet_model1, prenet_model2,  'train', 'y',
         **stft_param)

# f = open('mydata.pkl', 'rb')
# unpickler = cPickle.Unpickler(f)

# x_list = []
# while True:
#     try:
#         x = unpickler.load()
#         x_list.append(x)
#     except EOFError as e:
#         print('end of file')
#         break
# print('further operators')

# x_tr_stack = np.stack(x_list, axis=0)

# f = h5py.File('test_data.h5')  # HDF5 file
# d = f['/data/test_data']          # Pointer on on-disk array
# d.shape
# list(d.keys())
# d['block1_values']

# x = da.from_array(d, chunks=(1000, 1000))
