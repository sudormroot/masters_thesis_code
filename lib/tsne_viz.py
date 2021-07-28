import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import sys 

import random

from sklearn.manifold import TSNE


libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "." 
sys.path.append(libpath)


def sample_n_features(X, y, n, imgaug):
    sampled_features = []
    sampled_labels = []

    return sampled_features, sampled_labels


def plot_tsne_proj(features, labels, save_name = None):
      
    #colours = ["green", "gray", "brown", "blue", "red"]
 
    n_points = len(labels)

    features = features.copy().reshape(n_points, -1)

    n_class = len(set(labels))

    cmap = plt.cm.get_cmap("twilight_r", n_class)

    class_labels = list(set(labels))
    
    tsne = TSNE(n_jobs = 8, random_state = 42)
    tsne_embedding = tsne.fit_transform(features)
    
    #print("tsne_embedding.shape = ", tsne_embedding.shape)
    

    fig = plt.figure(figsize=(7, 7))
    lr = 150
    p = 50
    index = 0
    
    for i, Xi in enumerate(tsne_embedding):
        #print(Xi.shape)
        #colour = colours[i % 5]
        #colour = cmap(i)
        #plt.scatter(Xi[0], Xi[1], c = cmap(i))
        plt.scatter(Xi[0], Xi[1], colour = np.random.rand(3,1))
    
    
    #fig.legend(
    #    #bbox_to_anchor=(0.075, 0.061),
    #    loc = "lower left",
    #    ncol = 1,
    #    labels = class_labels,
    #)
    
    if save_name is not None:
        plt.savefig(
            "figures/" + save_name + ".svg", 
            bbox_inches="tight",)
    
    plt.show()
    plt.close()
