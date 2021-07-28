import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import sys 

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "." 
sys.path.append(libpath)

#from mbt_dataset import load_mbt_dataset_as_dataframe, load_mbt_dataset
#from mbt_split import split_mbt_dataset_as_dataframe, split_mbt_dataset


""" We plot the sample images from the dataset. 
"""

def plot_dataset_samples(df, nrows = 4, ncols = 4, subfig_height = 4, subfig_width = 4):
    
    
    
    
    """ Plotting the samples 
    
    """

    figsize = (ncols * subfig_width, nrows * subfig_height) 
    #figsize = (15, 15) 

    

    """ Sampling
    
    """
    dfs = df.sample(n = nrows * ncols, replace = False)
    
    Xs = np.array(dfs["X"], dtype = object)
    ys = np.array(dfs["y"], dtype = object)
    
    
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
            
            img = Xs[imgid]
            img = np.array(img, dtype = np.uint8) #print("img.shape", img.shape)
            
            label = ys[imgid]
            
            ax = axes_flat[row * ncols + col] 
            ax.set_xticks([])
            ax.set_yticks([])
            #ax.imshow(img, vmin = 0, vmax = 1) 
            ax.imshow(img) 
            ax.title.set_text(f"{label} #{imgid}")

    plt.show()
    plt.close()


#def plot_mbt_orig_dataset():
    # Loading dataset
#    df = load_mbt_dataset_as_dataframe()

#    plot_dataset_samples(df, nrows = 2, ncols = 4)



#def plot_train_dataset_samples():
#    X,y = load_mbt_dataset()
#    df_train, df_test = split_mbt_dataset_as_dataframe(X, y, split = 0.5)
#    plot_dataset_samples(df_train, nrows = 2, ncols = 4, subfig_height = 2.5)

#def plot_test_dataset_samples():
#    X,y = load_mbt_dataset()
#    df_train, df_test = split_mbt_dataset_as_dataframe(X, y, split = 0.5)
#    plot_dataset_samples(df_test, nrows = 2, ncols = 4, subfig_height = 2.5)
