#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:24:54 2023

@author: avk256
"""

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM, Reshape, Input, Lambda, Bidirectional, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
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
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from dask_ml.wrappers import Incremental
from scikeras.wrappers import KerasClassifier

import dask.dataframe as dd
import dask.array as da


import os

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# import librosa for analysing audio signals : visualize audio, display the spectogram

# import librosa for analysing audio signals : visualize audio, display the spectogram


# import wav for reading and writing wav files

# import IPython.dispaly for playing audio in Jupter notebook
# import IPython.display as ipd


tf.autograph.set_verbosity(3)

###############################################################################


def model_Dense(input_shape, n_clases=4, saved_file=None):
    """
    Dense model.

    Parameters
    ----------
    input_shape : TYPE
        DESCRIPTION.
    n_clases : TYPE, optional
        DESCRIPTION. The default is 4.
    saved_file : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    initializer = tf.keras.initializers.RandomNormal()
    initializer = tf.keras.initializers.RandomUniform()
    initializer = tf.keras.initializers.HeNormal()
    initializer = tf.keras.initializers.HeUniform()
    initializer = tf.keras.initializers.TruncatedNormal()
    initializer = tf.keras.initializers.GlorotNormal()
    initializer = tf.keras.initializers.GlorotUniform()
    initializer = tf.keras.initializers.LecunNormal()
    initializer = tf.keras.initializers.LecunUniform()

    optim = tf.keras.optimizers.Adam()
    optim = tf.keras.optimizers.Adadelta()
    optim = tf.keras.optimizers.Adagrad()
    optim = tf.keras.optimizers.Adamax()
    optim = tf.keras.optimizers.Nadam()
    optim = tf.keras.optimizers.SGD()
    optim = tf.keras.optimizers.RMSprop()
    optim = tf.keras.optimizers.Ftrl()

    initializer = tf.random_normal_initializer(0, 0.03)
    input_layer = Input(input_shape)

    y = Flatten()(input_layer)
    # y = Dense(512, kernel_initializer=initializer)(y)

    y = Dense(128, kernel_initializer=initializer)(y)
    y = Activation('relu')(y)
    # y = BatchNormalization()(y)
    # model.add(Dropout(0.25))

    # y = Dense(128, kernel_initializer=initializer)(
    #     y)  # input_shape=features.shape[1:]
    # y = Activation('relu')(y)
    # y = BatchNormalization()(y)

    # y = Dense(128, kernel_initializer=initializer)(
    #     y)  # input_shape=features.shape[1:]
    # y = Activation('relu')(y)
    # y = BatchNormalization()(y)
    y = Dropout(0.25)(y)
    y = Dense(n_clases, kernel_initializer=initializer)(y)
    out = Activation('softmax')(y)
    #sgd = optimizers.SGD(lr=0.1, decay=1e-3, momentum=1e-3)
    model = Model(inputs=[input_layer], outputs=out,)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optim,
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


def model_Conv(lr=0.01, momentum=0.9):
    """
    Conv model.

    Parameters
    ----------
    input_shape : TYPE
        DESCRIPTION.
    n_clases : TYPE, optional
        DESCRIPTION. The default is 4.
    saved_file : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    input_shape = (1057792, )
    n_clases = 4
    initializer = tf.random_normal_initializer(0, 0.03)

    initializer = tf.keras.initializers.RandomNormal()
    initializer = tf.keras.initializers.RandomUniform()
    initializer = tf.keras.initializers.HeNormal()
    initializer = tf.keras.initializers.HeUniform()
    initializer = tf.keras.initializers.TruncatedNormal()
    initializer = tf.keras.initializers.GlorotNormal()
    initializer = tf.keras.initializers.GlorotUniform()
    initializer = tf.keras.initializers.LecunNormal()
    initializer = tf.keras.initializers.LecunUniform()

    optimizers = tf.keras.optimizers.Adam()
    optimizers = tf.keras.optimizers.Adadelta()
    optimizers = tf.keras.optimizers.Adagrad()
    optimizers = tf.keras.optimizers.Adamax()
    optimizers = tf.keras.optimizers.Nadam()
    optimizers = tf.keras.optimizers.SGD()
    optimizers = tf.keras.optimizers.RMSprop()
    optimizers = tf.keras.optimizers.Ftrl()

    input_layer = Input(input_shape)
    model = Sequential()

    # 1. LAYER
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
              padding='Same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # # 2. LAYER
    # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same'))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))

    # model.add(MaxPool2D(pool_size=(2, 2)))

    # # 3. LAYER
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same'))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))

    # # 4. LAYER
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same'))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))

    model.add(MaxPool2D(pool_size=(2, 2)))

    # FULLY CONNECTED LAYER
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    # OUTPUT LAYER
    model.add(Dense(n_clases, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])

    return model


def model_Dense(lr=0.01, momentum=0.9):
    """
    Dense model.

    Parameters
    ----------
    input_shape : TYPE
        DESCRIPTION.
    n_clases : TYPE, optional
        DESCRIPTION. The default is 4.
    saved_file : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """
    input_shape = (1057792, )
    n_clases = 4
    saved_file = None
    initializer = tf.keras.initializers.RandomNormal()
    initializer = tf.keras.initializers.RandomUniform()
    initializer = tf.keras.initializers.HeNormal()
    initializer = tf.keras.initializers.HeUniform()
    initializer = tf.keras.initializers.TruncatedNormal()
    initializer = tf.keras.initializers.GlorotNormal()
    initializer = tf.keras.initializers.GlorotUniform()
    initializer = tf.keras.initializers.LecunNormal()
    initializer = tf.keras.initializers.LecunUniform()

    optim = tf.keras.optimizers.Adam()
    optim = tf.keras.optimizers.Adadelta()
    optim = tf.keras.optimizers.Adagrad()
    optim = tf.keras.optimizers.Adamax()
    optim = tf.keras.optimizers.Nadam()
    optim = tf.keras.optimizers.SGD()
    optim = tf.keras.optimizers.RMSprop()
    optim = tf.keras.optimizers.Ftrl()

    initializer = tf.random_normal_initializer(0, 0.03)
    input_layer = Input(input_shape)

    y = Flatten()(input_layer)
    # y = Dense(512, kernel_initializer=initializer)(y)

    y = Dense(128, kernel_initializer=initializer)(y)
    y = Activation('relu')(y)
    # y = BatchNormalization()(y)
    # model.add(Dropout(0.25))

    # y = Dense(128, kernel_initializer=initializer)(
    #     y)  # input_shape=features.shape[1:]
    # y = Activation('relu')(y)
    # y = BatchNormalization()(y)

    # y = Dense(128, kernel_initializer=initializer)(
    #     y)  # input_shape=features.shape[1:]
    # y = Activation('relu')(y)
    # y = BatchNormalization()(y)
    y = Dropout(0.25)(y)
    y = Dense(n_clases, kernel_initializer=initializer)(y)
    out = Activation('softmax')(y)
    #sgd = optimizers.SGD(lr=0.1, decay=1e-3, momentum=1e-3)
    model = Model(inputs=[input_layer], outputs=out,)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optim,
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


def df_reshape(x, y):
    """
    Reshape data for ANN.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    x_tr : TYPE
        DESCRIPTION.
    y_tr : TYPE
        DESCRIPTION.

    """
    x_tr = np.array(list(x))
    y_tr = np.array(y)

    x_tr = np.reshape(
        x_tr, (x_tr.shape[0], x_tr.shape[1], x_tr.shape[2], 1))
    y_tr = keras.utils.to_categorical(y_tr, num_classes=4, dtype='float32')
    return x_tr, y_tr


def plot_history(history):
    """
    Plot learning history.

    Parameters
    ----------
    history : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    acc = history.history['auc']
    val_acc = history.history['val_auc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, label='Training auc')
    plt.plot(epochs, val_acc, label='Validation auc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


###############################################################################
train_df = pd.read_pickle('train_data.pkl')

train_dd = dd.from_pandas(train_df, npartitions=10)

del train_df

x_train_dd = train_dd['ym'].to_dask_array()

x_train_dd.shape[1]

# train_dd = dd.read_csv('train_data.pkl')

# sample = train_df['ym'][0]

# print(train_df['y'].value_counts())


# # x_train, y_train = df_reshape(train_df['ym'], train_df['y'])

# x_train = np.array(list(train_df['ym']))
# y_train = np.array(train_df['y'])

# del train_df

# x_train = np.reshape(
#     x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
# y_train = keras.utils.to_categorical(y_train, num_classes=4, dtype='float32')


# # x_val, y_val = df_reshape(val_df['ym'], val_df['y'])

# val_df = pd.read_pickle('val_data.pkl')
# print(val_df['y'].value_counts())


# x_val = np.array(list(val_df['ym']))
# y_val = np.array(val_df['y'])

# del val_df

# x_val = np.reshape(
#     x_val, (x_val.shape[0], x_val.shape[1] * x_val.shape[2]))
# y_val = keras.utils.to_categorical(y_val, num_classes=4, dtype='float32')

# test_df = pd.read_pickle('test_data.pkl')
# print(test_df['y'].value_counts())

# x_test = np.array(list(test_df['ym']))
# y_test = np.array(test_df['y'])

# del test_df

# x_test = np.reshape(
#     x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# print(y_test[:10])

# # input_shape = (x_train.shape[1], x_train.shape[2])


# # Dask ML model


# niceties = dict(verbose=False)
# model = KerasClassifier(build_fn=model_Dense, lr=0.1,
#                         momentum=0.9, epochs=None, batch_size=None, verbose=0)


# # define the grid search parameters
# batch_size = [16, 32, 64]
# epochs = [5, 10]
# param_grid = dict(batch_size=batch_size, epochs=epochs)

# # search the grid
# grid = GridSearchCV(estimator=model,
#                     param_grid=param_grid,
#                     cv=3,
#                     verbose=2)  # include n_jobs=-1 if you are using CPU

# grid_result = grid.fit(x_train, y_train)


# # input_shape = (x_train.shape[1], x_train.shape[2])

# # model = model_Dense(input_shape)
# # # model = model_Conv(input_shape)
# # n_epochs = 2

# # # x = np.array(list(map(np.array, train_df['ym'])))
# # # y = np.array(train_df['y'])

# # model.compile(loss='categorical_crossentropy',
# #               optimizer=optimizers.Adam(lr=1e-4),
# #               metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])

# # history = model.fit(x_train, y_train,
# #                     batch_size=16,
# #                     epochs=n_epochs,
# #                     validation_data=(x_val, y_val),
# #                     verbose=1,)


# y_pred = model.predict(x_test)
# y_pred1 = np.argmax(y_pred, axis=1)

# # plot_history(history)

# target_names = ['class 0', 'class 1', 'class 2', 'class 3']
# print(classification_report(y_test, y_pred1, target_names=target_names))
