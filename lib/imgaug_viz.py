import numpy as np
import pandas as pd
import os
import sys 

import matplotlib.pyplot as plt

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "." 
sys.path.append(libpath)

from imgaug_model import build_augmentation_model


def plot_augmented_samples(X, y, nrows = 2, ncols = 4, subfig_width = 4, subfig_height = 4):

    #X,y = load_mbt_dataset()

    #X_train, y_train, X_test, y_test = split_mbt_dataset(X, y)

    img_augmenter = build_augmentation_model()


    figsize = (ncols * subfig_width, nrows * subfig_height)

    fig, axes = plt.subplots(nrows = nrows,
                             ncols = ncols,
                             figsize = figsize)

    """ plt.subplots_adjust(left = 0.02,
                    right = 0.97,
                    bottom = 0.05,
                    hspace = 0.1,
                    wspace = 0.1,
                    top = 0.95)
    """

    axes_flat = axes.flat


    for row in range(nrows):
        for col in range(ncols):

            imgid = row * ncols + col

            Xi = X[imgid, :, :, :]
            #Xi = X_train[3, :, :, :]
            Xi = Xi.copy()
            Xi = np.expand_dims(Xi, axis = 0)
            Xi = img_augmenter(Xi)
            #Xi = np.array(Xi, dtype = np.uint8)
            img = Xi[0, :, :, :]

            #img = X_train[imgid, :, :, :]
            img = np.array(img, dtype = np.uint8) #print("img.shape", img.shape)

            label = y[imgid]

            ax = axes_flat[row * ncols + col]
            ax.set_xticks([])
            ax.set_yticks([])
            #ax.imshow(img, vmin = 0, vmax = 1.0)
            ax.imshow(img)
            ax.title.set_text(f"{label} #{imgid}")

    plt.show()
    plt.close()




def plot_augmented_samples_for_one_image(X, y, nrows = 2, ncols = 4, subfig_width = 4, subfig_height = 4):
    
    #X,y = load_mbt_dataset()
    #X_train, y_train, X_test, y_test = split_mbt_dataset(X, y)
    
    img_augmenter = build_augmentation_model()
    
    
    figsize = (ncols * subfig_width, nrows * subfig_height) 
    
    fig, axes = plt.subplots(nrows = nrows,
                             ncols = ncols,
                             figsize = figsize)
    
    """ plt.subplots_adjust(left = 0.02,
                    right = 0.97,
                    bottom = 0.05,
                    hspace = 0.1,
                    wspace = 0.1,
                    top = 0.95)
    """

    axes_flat = axes.flat
    
    indices = list(range(len(y)))
    idx = np.random.choice(indices)
    candidate_image = X[idx, :, :, :]
    
    
    
    for row in range(nrows):
        for col in range(ncols):
            
            plotid = row * ncols + col
            
            Xi = candidate_image.copy()
            Xi = np.expand_dims(Xi, axis = 0)
            Xi = img_augmenter(Xi)
            #Xi = np.array(Xi * 255, dtype = np.uint8)
            img = Xi[0, :, :, :]
    
            #img = X_train[imgid, :, :, :]
            img = np.array(img, dtype = np.uint8) #print("img.shape", img.shape)
            
            label = y[idx]
            
            ax = axes_flat[row * ncols + col] 
            ax.set_xticks([])
            ax.set_yticks([])
            #ax.imshow(img, vmin = 0, vmax = 1.0) 
            ax.imshow(img) 
            ax.title.set_text(f"{label} #{plotid}")

    plt.show()
    plt.close()
