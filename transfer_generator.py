#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:59:28 2023

@author: avk256
"""
# os.chdir(./)
# print("Current working directory: {0}".format(os.getcwd()))

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools

import os

import seaborn as sns
import plotly.express as px

# import librosa for analysing audio signals : visualize audio, display the spectogram
import librosa
import soundfile as sf

# import librosa for analysing audio signals : visualize audio, display the spectogram
import librosa.display


# import wav for reading and writing wav files
import wave

# import IPython.dispaly for playing audio in Jupter notebook
# import IPython.display as ipd

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM, Reshape, Input, Lambda, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import tensorflow.keras as keras

import tensorflow_addons as tfa

import random
import time
import re
import itertools
from sklearn.model_selection import StratifiedShuffleSplit
import pdb
import line_profiler
profile = line_profiler.LineProfiler()

###############################################################################


def data_splitting(X, y):

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
    print("File " + sounds_status[cat][sound][0] + " loaded")
    filename = (sounds_status[cat][sound][0].split(
        '/')[-1]).split('__')[0] + '.wav'

    for segment in range(1, len(sounds_status[cat][sound])):
        y, sr = librosa.load(sounds_status[cat][sound][segment])
        print("File " + sounds_status[cat][sound][segment] + " loaded")
        x = np.append(x, y)
#     sf.write(filename, x, sr, subtype='PCM_24')
    return x, sr


def sound_load(file_name, sounds_status):

    indexes = find_sound(sounds_status, file_name)
    cat, sound = indexes[0].split(' ')
    cat, sound = int(cat), int(sound)
    x, sr = wav_join(sounds_status, cat, sound)

    return x, sr


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


class DataAudioGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 df_table,
                 train_folder,
                 target,
                 sounds_status,
                 batch_size,
                 n_fft,        # stft params
                 win_length,   # stft params
                 hop_length,   # stft params
                 w_lenght=6,  # duration time for every clip audio
                 stride=3,   # Only for test pourpouse, to cut audio into pises of w_lenght with stride stride jeje
                 samplerate=44000,
                 childrens=2,  # number of synthetic audios for every original
                 shuffle=False,
                 num_classes=None,
                 noise_var=0.0025,
                 random_state=None,
                 train_clip_folder=None,
                 pretrain_model_name='YAMNET',
                 n_inputs='Xym',
                 mode="TRAIN",
                 model1=[],
                 model2=[],):

        assert batch_size % (
            childrens+1) == 0, 'batch_size must be a multiple of (children+1)'
        self.batch_sizes = batch_size//(childrens+1)
        self.childrens = childrens
        self.stride = stride
        self.mode = mode
        self.train_folder = train_folder
        self.train_clip_folder = train_clip_folder
        self.w_lenght = w_lenght
        self.samplerate = samplerate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        # "win_length" :win_length,
        self.stft_param = {"n_fft": n_fft,  "hop_length": hop_length}
        self.df_table = df_table
        self.target = target
        self.random_state = random_state
        self.indices = df_table.index.tolist()
        self.labels = df_table[self.target] if mode == "TRAIN" or "VAL" else []
        self.num_classes = num_classes
        self.noise_var = noise_var
        self.shuffle = shuffle
        self.noise_v = [np.stack([random.uniform(-self.noise_var, self.noise_var) for _ in range(self.samplerate*self.w_lenght)]),
                        np.stack([random.gauss(0, self.noise_var)
                                 for _ in range(self.samplerate*self.w_lenght)]),
                        ]
        self.prenet_model1 = model1
        self.prenet_model2 = model2
        self.pretrain_model_name = pretrain_model_name
        self.sounds_status = sounds_status
        self.n_inputs = n_inputs
        self.on_epoch_end()
        self.__info()

    def __info(self):
        #         print('__info')
        #         pdb.set_trace()
        print("Audio generator=====>")
        print(
            "We have {:4d} original audios - {}".format(len(self.indices), self.mode))
        if self.mode == "TRAIN":
            print("We'll generate and add {:4d} synthetic audios - {}".format(
                len(self.indices)*self.childrens, self.mode))

    def __len__(self):
        #         print('__len')
        #         pdb.set_trace()
        return len(self.indices) // self.batch_sizes

    @profile
    def __getitem__(self, index):
        #         print('__getitem__')
        #         pdb.set_trace()
        self.batch_sizes = 1 if self.mode == "TEST" else self.batch_sizes
        index = self.index[index *
                           self.batch_sizes:(index + 1) * self.batch_sizes]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        #         print('on_epoch_end')
        #         pdb.set_trace()
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    @profile
    def __get_data(self, batch):
        #         print('__get_data')
        #         pdb.set_trace()
        X = []
        Xym = []
        Xvgg = []
        y = []
        # print(batch)
        if self.mode == "TRAIN" or "VAL":
            for i, id in enumerate(batch):
                y_yi = self.labels[id]
                hotencode_yi = keras.utils.to_categorical(
                    y_yi, num_classes=self.num_classes, dtype='float32')
                signal, feature_yamnet = self.extract_audio_clip(
                    id)
                spec_db = self.spec_transp_scal(signal)

                X.append(spec_db)       # original X
                Xym.append(feature_yamnet)
                # Xvgg.append(feature_vggih)
                y.append(hotencode_yi)  # original Y
#                 for i_aum in range(1,self.childrens+1):
#         '            random_base = self.random_state + int(time.time())
#                     random.seed(random_base)
#                     signal = self.noise_signal(signal)
#                     spec_db = self.spec_transp_scal(signal)
#                     X.append(spec_db)       # original X
#                     Xym.append(self.extract_featur(signal))
#                     Xvgg.append(self.extract_featur(signal, "VGGISH"))
#                     y.append(hotencode_yi)  # original Y
#             print(self.mode)
#             print(self.n_inputs)
            if self.n_inputs == 'all':
                return (np.expand_dims(np.stack(X), axis=3), np.stack(Xym), np.stack(Xvgg)), np.stack(y)
            if self.n_inputs == 'Xym':
                #                 print('return (np.stack(Xym)), np.stack(y)')
                return (np.stack(Xym)), np.stack(y)
            if self.n_inputs == 'Xvgg':
                return (np.stack(Xvgg)), np.stack(y)
            if self.n_inputs == 'X':
                return (np.expand_dims(np.stack(X), axis=3)), np.stack(y)

        elif self.mode == "TEST":
            for i, id in enumerate(batch):
                signal, feature_prenet, feature_vggih = self.extract_audio_clip(
                    id)
                spec_db = np.stack([self.spec_transp_scal(signal_i)
                                   for signal_i in signal])
            return (np.expand_dims(spec_db, axis=3), feature_prenet, feature_vggih), []

    @profile
    def extract_audio_clip(self, index):
        #         print('extract_audio_clip')
        #         pdb.set_trace()
        # train_clip_folder
        record_name = self.df_table.loc[index]["file name"]
        signal, srate = self.sound_load(record_name)

        # self.extract_featur(signal, "VGGISH")
        return signal, self.extract_featur(signal)

    @profile
    def extract_featur(self, signal, model="YAMNET"):
        #         print('extract_featur')
        #         pdb.set_trace()
        if model == "VGGISH":
            #             return self.prenet_model1(signal)
            return []
        elif model == "YAMNET":
            return self.prenet_model2(signal)[1]

    @profile
    def spec_transp_scal(self, signal):
        #         print('spec_transp_scal')
        #         pdb.set_trace()
        # librosa.amplitude_to_db(abs(librosa.stft(signal, **self.stft_param)))
        signal = librosa.feature.melspectrogram(
            y=signal, sr=self.samplerate, **self.stft_param)
        # transpose because we going to reshape inside the model and use a LSTM to extract time features
        signal = np.transpose(signal[-int(len(signal)/4):])
        return ((signal - 1.0)/90.0)  # min max scale

    @profile
    def noise_signal(self, signal):
        def manipulate(data, noise_factor):
            noise = np.random.randn(len(data))
            augmented_data = data + noise_factor * noise
            # Cast back to same data type
            augmented_data = augmented_data.astype(type(data[0]))
            return augmented_data
#         print('noise_signal')
#         pdb.set_trace()
        noise_factor = random.uniform(0.0001, 0.001)
        signal = manipulate(signal, noise_factor)

        return signal

    def find_sound(self, filename):
        return ["{} {}".format(index1, index2) for index1, value1 in enumerate(self.sounds_status) for index2, value2 in enumerate(value1) for index3, value3 in enumerate(value2) if value3.split('/')[-1][:-14] == filename[:-4]]

    def wav_join(self, cat, sound):
        #         print('wav_join')
        #         pdb.set_trace()
        x, sr = librosa.load(self.sounds_status[cat][sound][0])
#         print("File " + self.sounds_status[cat][sound][0] + " loaded")
        filename = (self.sounds_status[cat][sound][0].split(
            '/')[-1]).split('__')[0] + '.wav'

        for segment in range(1, len(self.sounds_status[cat][sound])):
            y, sr = librosa.load(self.sounds_status[cat][sound][segment])
#             print("File " + self.sounds_status[cat][sound][segment] + " loaded")
            x = np.append(x, y)
    #     sf.write(filename, x, sr, subtype='PCM_24')
        return x, sr

    def sound_load(self, file_name):
        #         print('sound_load')
        #         pdb.set_trace()
        indexes = self.find_sound(file_name)
        cat, sound = indexes[0].split(' ')
        cat, sound = int(cat), int(sound)
        x, sr = self.wav_join(cat, sound)

        return x, sr


###############################################################################

sound_list = []
for dirname, _, filenames in os.walk('./data/data/'):
    for filename in filenames:
        # print(os.path.join(dirname, filename))
        sound_list.append(os.path.join(dirname, filename))

data = "./data/data/sound_files/"
table_df = pd.read_csv('./data/data/all_data_updated.csv')
wav_files = []
for path, subdirs, files in os.walk(data):
    for name in files:
        wav_files.append(os.path.join(path, name))

print(table_df.columns)
queen_status_dict = {0: 'original queen', 1: 'not present',
                     2: 'present and rejected', 3: 'present and newly accepted'}


no_files = []
sounds_status = [0]*4
files_status = [0]*4
for i in range(0, 4):
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


for i in range(len(table_dur_df)):
    permut = list(itertools.permutations(range(table_dur_df['durat'][i]), 3))
    sample_permut = np.random.permutation(permut)[0:10]
    table_dur_df.at[i, 'samples'] = " ".join(list(map(str, sample_permut)))

table_dur_df = table_dur_df[table_dur_df['durat'] == 6]
table_dur_df = table_dur_df.reset_index()

X_train, y_train, X_test, y_test, X_val, y_val = data_splitting(
    table_dur_df['file name'], table_dur_df['queen status'])
print(len(X_train))
print(len(X_test))
print(len(X_val))
print(X_val)
print(y_val)

df_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
print(df_train)

df_test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
print(df_test)

df_val = pd.concat([pd.DataFrame(X_val), pd.DataFrame(y_val)], axis=1)
print(df_val)


x, sr = sound_load('2022-06-08--15-51-41_1.raw', sounds_status)
# from IPython.display import Audio
# Audio(x, rate=sr)

# prenet_model1 = hub.load('https://tfhub.dev/google/vggish/1')
prenet_model2 = hub.load('https://tfhub.dev/google/yamnet/1')
prenet_model1 = prenet_model2


train_folder = "./data/data/sound_files/"
param_generator_train = {
    "df_table": df_train,
    "train_folder": train_folder,
    "target": 'queen status',
    "train_clip_folder": None,
    "batch_size": 44,
    "n_fft": 1100,        # stft params
    "win_length": 1100,   # stft params
    "hop_length": 512,   # stft params
    "samplerate": 44000,
    "w_lenght": 3,  # Signal window length to analyze and then feed to our model
    "childrens": 3,  # Number of augmented audios for every original audio
    "shuffle": True,
    "num_classes": 4,
    "noise_var": 0.0025,
    "random_state": 10,
    "sounds_status": sounds_status,
    "n_inputs": 'Xym',
    "model1": prenet_model1,
    'model2': prenet_model2,
    # "pretrain_model_name": 'VGGISH',
}
param_generator_val = {
    "df_table": df_val,
    "train_folder": train_folder,
    "target": 'queen status',
    "train_clip_folder": None,
    "batch_size": 44,
    "n_fft": 1100,        # stft params
    "win_length": 1100,   # stft params
    "hop_length": 512,   # stft params
    "samplerate": 44000,
    "w_lenght": 3,  # Signal window length to analyze and then feed to our model
    "childrens": 1,  # Number of augmented audios for every original audio
    "shuffle": True,
    "num_classes": 4,
    "noise_var": 0.0025,
    "random_state": 10,
    "sounds_status": sounds_status,
    "n_inputs": 'Xym',
    "mode": "VAL",
    "model1": prenet_model1,
    'model2': prenet_model2,
    # "pretrain_model_name": 'VGGISH',
}
param_test = {
    "df_table": df_test,
    "train_folder": train_folder,
    "target": 'queen status',
    "batch_size": 4,
    "n_fft": 1100,        # stft params
    "win_length": 1100,   # stft params
    "hop_length": 512,   # stft params
    "samplerate": 44000,
    "w_lenght": 3,  # Signal window length to analyze and then feed to our model
    "stride": 1,
    "childrens": 1,  # Number of augmented audios for every original audio
    "shuffle": False,
    "num_classes": 4,
    "noise_var": 0.0025,
    "random_state": 10,
    "mode": "TEST",
    "sounds_status": sounds_status,
    "n_inputs": 'Xym',
    "model1": prenet_model1,
    'model2': prenet_model2,
    # "pretrain_model_name": 'VGGISH',
}


len(df_train)
len(df_val)
len(df_test)

train_datagen = DataAudioGenerator(**param_generator_train)
val_datagen = DataAudioGenerator(**param_generator_val)
test_datagen = DataAudioGenerator(**param_test)

sample_item_data = train_datagen.__getitem__(1)[0]
input_shape = sample_item_data.shape[1:]


print("Model Input size: {}".format(input_shape))

print(np.shape(sample_item_data))


model = model_Dense(input_shape)
n_epochs = 10

history = model.fit(train_datagen,
                    #steps_per_epoch = 8,   #
                    batch_size=32,          #
                    epochs=n_epochs,
                    #                     initial_epoch = n_epochs,
                    #                     callbacks=[tboard_callback,model_checkpoint,earlyStopping],
                    #                     callbacks=[earlyStopping],
                    validation_data=val_datagen,
                    workers=-1,
                    verbose=1,
                    )
