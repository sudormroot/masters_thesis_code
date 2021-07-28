import numpy as np
import pandas as pd
import os
import sys 

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "."
sys.path.append(libpath)

from tensorflow import keras


def split_keras_dataset_to_xy(ds):
    X_ = ds.map(lambda x, y: x)
    y_ = ds.map(lambda x, y: y)
    
    X = [Xi for Xi in X_]
    y = [yi for yi in y_]

    X = np.array(X)
    y = np.array(y)

    # the original shape is in (n_batches, batch_size, width, height, n_chans)
    data_shape = X.shape[2:]
    print("data_shape = ", data_shape)

    X = X.reshape(-1, *data_shape)
    y = y.reshape(-1)

    print("X.shape = ", X.shape)
    print("y.shape = ", y.shape)

    return X, y
