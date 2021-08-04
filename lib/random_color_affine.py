# The code is from:
# https://keras.io/examples/vision/semisupervised_simclr/

import numpy as np
import pandas as pd
import os
import sys 


libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "." 
sys.path.append(libpath)


import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing


# We make it working with keras model save and load
class CustomisedRandomColorAffineLayer(keras.layers.Layer):

    def __init__(self, 
                 brightness = 0, 
                 jitter = 0, 
                 #name = 'CustomisedRandomColorAffineLayer',
                 **kwargs):

        self.brightness = brightness
        self.jitter = jitter
        
        super(CustomisedRandomColorAffineLayer, self).__init__(**kwargs)

    def get_config(self):
        
        config = super().get_config()


        config['brightness'] = self.brightness
        config['jitter'] = self.jitter

        print("config = ", config)

        return config


    def call(self, images, training = True):
        if training:
            batch_size = tf.shape(images)[0]

            # For each pixels, we randomly adjust their brightnesses.
            brightness_scales = 1 + tf.random.uniform(
                            (batch_size, 1, 1, 1), 
                            minval = -self.brightness, 
                            maxval = self.brightness
                        )

            # Different for all colors
            jitter_matrices = tf.random.uniform(
                            (batch_size, 1, 3, 3), 
                            minval = -self.jitter, 
                            maxval = self.jitter
                        )

            color_transforms = (
                            tf.eye(3, batch_shape = [batch_size, 1]) * brightness_scales
                                + jitter_matrices
                        )

            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)

        return images


