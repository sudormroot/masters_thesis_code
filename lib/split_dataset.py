import numpy as np
import pandas as pd
import os
import sys 

#libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep + "lib"
libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "."
sys.path.append(libpath)

DS_DEFAULT_PIXEL_SPLIT_RATIO = 0.5

def split_dataset_by_pixel(X, y, split = DS_DEFAULT_PIXEL_SPLIT_RATIO):
    X_train = []
    y_train = []
    
    X_test = []
    y_test = []
    
    
    for Xi, yi in zip(X, y):
        Xi = np.array(Xi)
        #print(Xi.shape)
        #print(yi)
        
        height = Xi.shape[0]
        
        position = int(split * height)

        Xi_train = Xi[0:position, :, :]
        Xi_test = Xi[position:height, :, :]
        
        #print(Xi_train.shape)
        #print(Xi_test.shape)
        
        yi_train = yi
        yi_test = yi
        
        X_train.append(Xi_train)
        y_train.append(yi_train)
        
        X_test.append(Xi_test)
        y_test.append(yi_test)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print()
    print("Splitting dataset into training and testing datasets ...")
    print(f"size of training dataset : {len(y_train)}")
    print(f"size of testing dataset : {len(y_test)}")
    print(f"training dataset shape: {X_train.shape}")
    print(f"testing dataset shape: {X_test.shape}")
    #print()
    
    return X_train, y_train, X_test, y_test


def split_dataset_by_pixel_as_dataframe(X, y, split = DS_DEFAULT_PIXEL_SPLIT_RATIO):

    X_train, y_train, X_test, y_test = split_dataset_by_pixel(X, y, split)

    data_train = {"X": list(X_train),
                  "y": list(y_train)}

    data_test = {"X": list(X_test),
                  "y": list(y_test)}

    df_train = pd.DataFrame(data_train)

    df_test = pd.DataFrame(data_test)

    #print("X_train.shape = ", np.array(X_train).shape)

    return df_train, df_test

#def test_split_mbt_dataset():
#    X,y = load_mbt_dataset()
#    df_train, df_test = split_mbt_dataset_as_dataframe(X, y, split = DS_DEFAULT_PIXEL_SPLIT_RATIO)



