import numpy as np
import pandas as pd
import os
import sys 

import matplotlib.pyplot as plt

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "." 
sys.path.append(libpath)

from imgaug_model import (CustomisedTrainImageAugmenter, CustomisedUtilImageAugmenter)

"""
def plot_augmented_samples_train(dataset, n_samples, strong_imgaug_params, weak_imgaug_params):
        
    images = next(iter(dataset))[0][:n_samples]
       
    """augmented_images = zip(
        images,
        build_train_image_augmenter(**weak_imgaug_params)(images),
        build_train_image_augmenter(**strong_imgaug_params)(images),
        build_train_image_augmenter(**strong_imgaug_params)(images),
    )
    """

    augmented_images = zip(
        images,
        CustomisedTrainImageAugmenter(**weak_imgaug_params)(images),
        CustomisedTrainImageAugmenter(**strong_imgaug_params)(images),
        CustomisedTrainImageAugmenter(**strong_imgaug_params)(images),
    )


    row_titles = [
        "Original:",
        "Weak augmentation:",
        "Strong augmentation:",
        "Strong augmentation:",
    ]
    
    plt.figure(figsize=(n_samples * 2.2, 4 * 2.2), dpi = 100)
    
    for column, image_row in enumerate(augmented_images):
        for row, image in enumerate(image_row):
            plt.subplot(4, n_samples, row * n_samples + column + 1)
            plt.imshow(image, vmin=0, vmax=1)
            if column == 0:
                plt.title(row_titles[row], loc="left")
            plt.axis("off")

    plt.tight_layout()
    plt.show()
"""



def plot_augmented_samples(*, X, y, img_augmenter, nrows = 2, ncols = 4, subfig_width = 4, subfig_height = 4):

    #X,y = load_mbt_dataset()

    #X_train, y_train, X_test, y_test = split_mbt_dataset(X, y)

    #img_augmenter = CustomisedUtilImageAugmenter()


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




def plot_augmented_samples_for_one_image_util(*, X, y, img_augmenter, nrows = 2, ncols = 4, subfig_width = 4, subfig_height = 4):
    
    #X,y = load_mbt_dataset()
    #X_train, y_train, X_test, y_test = split_mbt_dataset(X, y)
    
    #img_augmenter = CustomisedUtilImageAugmenter()
    
    
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
