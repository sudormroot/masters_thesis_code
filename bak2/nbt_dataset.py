import numpy as np
import pandas as pd
import os
import sys 

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "."
sys.path.append(libpath)


NBT_DS_IMAGE_SIZE = (512, 512)

def load_normalised_brodatz_dataset(image_size = NBT_DS_IMAGE_SIZE):
    dataset_path = "datasets" + os.path.sep + "Brodatz_Normalised_Texture"
    
    print()
    print(f"Loading dataset from path: {dataset_path} ...")
    print("image size: ", image_size)

    X_train = np.load(dataset_path + os.path.sep + "BrNoRo_X_encoder.npy")
    y_train = np.load(dataset_path + os.path.sep + "BrNoRo_y_encoder.npy")

    X_train = X_train.astype('float32')
    X_train /= 255 
    X_train -= 0.5

    return X_train, y_train
