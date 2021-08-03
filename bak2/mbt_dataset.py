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


MBT_DS_IMAGE_SIZE = (512, 512)

def load_mbt_dataset(image_size = MBT_DS_IMAGE_SIZE):
    
    dataset_path = "datasets" + os.path.sep + "Multiband_Brodatz_Texture"
    
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
    X = X.astype('float32')
    #X /= 255.0
    #X -= 0.5
    
    return X, y 


def load_mbt_dataset_as_dataframe():
    X, y = load_mbt_dataset()
    
    data = {"X": list(X),
            "y": list(y)}
    
    df = pd.DataFrame(data) 
    
    return df  


def save_mbt_as_numpy():
    X, y = load_mbt_dataset()

    np.save("mbt_X.npy", X)
    np.save("mbt_y.npy", y)

def test_load_mbt_dataset():
    X,y = load_mbt_dataset()
    print()
    print("num_of_samples: ", len(y))
    print("shape: ", X.shape)
    #print(X[0])


def test_load_mbt_dataset_as_dataframe():
    df = load_mbt_dataset_as_dataframe()
    print()
    print("num_of_samples: ", len(df))


