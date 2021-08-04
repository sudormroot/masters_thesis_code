import numpy as np
import pandas as pd
import os
import sys 

import tensorflow as tf
from tensorflow import keras

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "." 
sys.path.append(libpath)

from random_color_affine import CustomisedRandomColorAffineLayer

# refer: https://keras.io/examples/vision/semisupervised_simclr/
#
# Random color affine transformation is crucial to avoid trivial solution of merely
# learning classification from color histogram information!
#
def build_train_image_augmenter( input_shape, 
                                    min_area, 
                                    brightness, 
                                    jitter):
    
    print()
    print("input_shape = ", input_shape)
    print("min_area = ", min_area)
    print("brightness = ", brightness)
    print("jitter = ", jitter)

    zoom_factor = 1.0 - tf.sqrt(min_area)
    
    return keras.Sequential(
        [
            #keras.layers.InputLayer(input_shape=(image_size, image_size, image_channels)),
            keras.layers.InputLayer(input_shape = input_shape),
            #preprocessing.Rescaling(1 / 255),
            keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            keras.layers.experimental.preprocessing.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            keras.layers.experimental.preprocessing.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            CustomisedRandomColorAffineLayer(   brightness = brightness, 
                                                jitter = jitter, 
                                                name = "CustomisedRandomColorAffineLayer"),
        ],
        name = "train_image_augmenter"
    )


# this augmentation model is used for tools
def build_util_image_augmenter( 
                             input_shape = (256, 512, 3), 
                             output_shape = (128, 128, 3),
                             seed = 21):
    
    
    return keras.Sequential([
              #keras.layers.InputLayer(input_shape = input_shape), 
              #keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
              #keras.layers.experimental.preprocessing.Normalization(),
              #keras.layers.experimental.preprocessing.RandomFlip(seed = seed),
              #keras.layers.experimental.preprocessing.RandomZoom(0.5),
              keras.layers.experimental.preprocessing.RandomRotation(0.5, seed = seed),
              keras.layers.experimental.preprocessing.RandomCrop(output_shape[0], 
                                                                 output_shape[1],
                                                                 seed = seed)
        
             ],
             name = "util_image_augmenter")
    
    



