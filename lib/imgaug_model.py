import numpy as np
import pandas as pd
import os
import sys 

from tensorflow import keras

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "." 
sys.path.append(libpath)

def build_augmentation_model(*, 
                             input_shape = (256, 512, 3), 
                             output_shape = (128, 128, 3),
                             rand_seed = 21):
    
    
    print()
    print("Building augmentation model ...")
    print(f"input_shape = {input_shape}")
    print(f"output_shape = {output_shape}")
    
    """
    inputs = keras.Input(shape = input_shape)
    
    #x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = keras.layers.experimental.preprocessing.Normalization()(inputs)
    
    #x = keras.layers.experimental.preprocessing.RandomFlip()(x)

    #x = keras.layers.experimental.preprocessing.RandomZoom(0.5)(x)

    x = keras.layers.experimental.preprocessing.RandomRotation(0.5)(x)
    
    #x = keras.layers.experimental.preprocessing.RandomContrast(0.5)(x)
    
    #print("input_shape = ", input_shape)
    #print("output_shape = ", output_shape)
    
    outputs = keras.layers.experimental.preprocessing.RandomCrop(output_shape[0], output_shape[1])(x)
      
    
    model = keras.Model(inputs = inputs, outputs = outputs, name = "image_augmenter")
    """
    
    model = keras.Sequential([
              #keras.layers.InputLayer(input_shape = input_shape), 
              #keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
              #keras.layers.experimental.preprocessing.Normalization(),
              keras.layers.experimental.preprocessing.RandomFlip(seed = rand_seed),
              #keras.layers.experimental.preprocessing.RandomZoom(0.5),
              keras.layers.experimental.preprocessing.RandomRotation(0.5, seed = rand_seed),
              keras.layers.experimental.preprocessing.RandomCrop(output_shape[0], 
                                                                 output_shape[1],
                                                                 seed = rand_seed)
        
             ])
    
    
    return model




