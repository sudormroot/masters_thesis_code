import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import os
import sys 

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "." 
sys.path.append(libpath)

import tensorflow as tf
from tensorflow import keras
import keras.backend as K


def build_simple_embedding_model_old(*, img_size = (128, 128, 3), embedding_dim = 10):
    model = keras.Sequential()


    #model.add(keras.Input(shape = img_size))
    model.add(keras.layers.Convolution2D(32, (11, 11), activation='relu',
                            input_shape=img_size))
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Convolution2D(32, (11, 11), activation='relu'))
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    #model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Convolution2D(32, (11, 11), activation='relu'))
    model.add(keras.layers.AveragePooling2D())


    model.add(keras.layers.Flatten())
    model.add(keras.layers.BatchNormalization())

    # model.add(Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    # tanh => output is in [-1, 1]^embedding_dim
    model.add(keras.layers.Dense(embedding_dim, activation='tanh'))

    print("embedding_dim = ", embedding_dim)

    return model


def build_simple_embedding_model(*, img_size = (128, 128, 3), embedding_dim = 10):
    """A small convolutional model. Its input is an image and output is an
    embedding, ie a vector. We don't compile or add a loss since this
    model will become a component in the complete model below.
    """

    model = keras.Sequential()
    #model.add(keras.Input(shape = img_size))
    model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu',
                            input_shape=img_size))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Convolution2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.GlobalMaxPooling2D())
    model.add(keras.layers.Dropout(0.1))
    # model.add(Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    # tanh => output is in [-1, 1]^embedding_dim
    model.add(keras.layers.Dense(embedding_dim, activation='tanh'))
    return model



