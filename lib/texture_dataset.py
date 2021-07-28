import numpy as np
import pandas as pd
import os
import sys 

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "."
sys.path.append(libpath)

from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


import matplotlib.pyplot as plt


TEXTURE_DS_IMAGE_SIZE = (512, 512)


def get_texture_subdirs():
    dataset_root = "datasets_orig"

    subdirs = [d.path for d in os.scandir(dataset_root) 
                        if d.is_dir(follow_symlinks = False) and 
                            not d.name.startswith(".") and 
                            not d.name.endswith("_npy")]

    subdirs = [s.rsplit(r"/", 1)[1] for s in subdirs]

    return subdirs

def load_texture_dataset(*, image_size = TEXTURE_DS_IMAGE_SIZE, subdir):
    
    assert subdir is not None, "The subdir must be specified."

    #dataset_path = "datasets_orig" + os.path.sep + "Multiband_Brodatz_Texture"
    dataset_path = "datasets_orig" + os.path.sep + subdir

    assert os.path.exists(dataset_path), f"{dataset_path} does not exist."

    
    print()
    print(f"Loading dataset from path: {dataset_path} ...")
    print("image size: ", image_size)
    
    X = [] 
    y = []
    

    # Get the subdirs for given dataset path

    files = []
    
    suffixes = (".tif", ".gif", ".png", ".jpg")
    
    imgfiles = [d.path for d in os.scandir(dataset_path) 
                    if d.is_file(follow_symlinks = False) and 
                        not d.name.startswith(".") and
                        d.name.lower().endswith(suffixes) ]
    
    #print(imgfiles)
    
    for p in sorted(imgfiles):
        #print(p)

        
        # Loading images and pre-processing images
        img = load_img( path = p,
                        grayscale = False, 
                        color_mode='rgb', 
                        target_size = image_size)
        
        # Converting images into numpy arraries.
        img = img_to_array(
                        img = img,
                        data_format = "channels_last",
                        dtype = np.double)
        
                        
        filename, file_extension = os.path.splitext(p)
        
        label = os.path.basename(filename)
        
        #print(label)

        X.append(img)
        y.append(label)
        
    print(f"Total {len(y)} images are loaded.")    
    #print()
    
    X = np.array(X)
    y = np.array(y)
    #X = X.astype('float32')
    
    return X, y 


def load_texture_dataset_as_dataframe(*, image_size = TEXTURE_DS_IMAGE_SIZE, subdir):

    X, y = load_texture_dataset(image_size = image_size, subdir = subdir)
    
    data = {"X": list(X),
            "y": list(y)}
    
    df = pd.DataFrame(data) 
    
    return df  


def save_texture_dataset_as_numpy(*, image_size = TEXTURE_DS_IMAGE_SIZE, subdir, save_prefix):

    X, y = load_texture_dataset(image_size = image_size, subdir = subdir)

    np.save(f"{save_prefix}_X.npy", X)
    np.save(f"{save_prefix}_y.npy", y)

def test_load_texture_dataset(*, subdir):
    
    X,y = load_texture_dataset(subdir = subdir)

    print()
    print("num_of_samples: ", len(y))
    print("shape: ", X.shape)
    #print(X[0])


def test_load_texture_dataset_as_dataframe(*, subdir):
    df = load_texture_dataset_as_dataframe(subdir)
    print()
    print("num_of_samples: ", len(df))


