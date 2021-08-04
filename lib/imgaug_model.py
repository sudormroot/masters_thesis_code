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
class CustomisedTrainImageAugmenter(keras.layers.Layer):
    def __init__(self, 
                 image_shape,
                 min_area,
                 brightness,
                 jitter,
                 name = 'train_imgaug_layer',
                 **kwargs):

        self.image_shape = image_shape
        self.min_area = min_area
        self.brightness = brightness
        self.jitter = jitter


        #self.zoom_factor = 1.0 - tf.sqrt(min_area)
        self.zoom_factor = 1.0 - np.sqrt(min_area)

        self.model = keras.Sequential(
           [
            keras.layers.InputLayer(input_shape = image_shape),
            #preprocessing.Rescaling(1 / 255),
            keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            keras.layers.experimental.preprocessing.RandomTranslation(self.zoom_factor / 2, self.zoom_factor / 2),
            keras.layers.experimental.preprocessing.RandomZoom((-self.zoom_factor, 0.0), (-self.zoom_factor, 0.0)),
            CustomisedRandomColorAffineLayer(   brightness = brightness, 
                                                jitter = jitter, 
                                                name = "color_affine_for_train"),
            ],
            name = "train_imgaug_model"
          )

        super(CustomisedTrainImageAugmenter, self).__init__(name = name, **kwargs)

    def get_config(self):
        config = super().get_config()
        
        config['image_shape'] = self.image_shape
        config['min_area'] = self.min_area
        config['brightness'] = self.brightness
        config['jitter'] = self.jitter
        config['zoom_factor'] = self.zoom_factor
        config['model'] = self.model

        #print(config)

        return config

    def call(self, images, training = True):
        if training:
            outputs = self.model(images)
        else:
            outputs = images

        return outputs

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class CustomisedUtilImageAugmenter(keras.layers.Layer):
    def __init__(self,
                 image_shape = (256, 512, 3),
                 target_shape = (128, 128, 3),
                 seed = 21,
                 name = 'util_imgaug_layer',
                 **kwargs):

        self.image_shape = image_shape
        self.target_shape = target_shape
        self.seed = seed

        self.model = keras.Sequential([
              keras.layers.InputLayer(input_shape = self.image_shape), 
              #keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
              #keras.layers.experimental.preprocessing.Normalization(),
              #keras.layers.experimental.preprocessing.RandomFlip(seed = seed),
              #keras.layers.experimental.preprocessing.RandomZoom(0.5),
              keras.layers.experimental.preprocessing.RandomRotation(0.5, seed = self.seed),
              keras.layers.experimental.preprocessing.RandomCrop(self.target_shape[0],
                                                                 self.target_shape[1],
                                                                 seed = self.seed)

             ],
             name = "util_imgaug_model")

        super(CustomisedUtilImageAugmenter, self).__init__(name = name, **kwargs)

    def get_config(self):
        config = super().get_config()
        
        config['image_shape'] = self.image_shape
        config['target_shape'] = self.target_shape
        config['seed'] = self.seed
        config['model'] = self.model

        return config

    def call(self, images, training = False):
        outputs = self.model(images)
        return outputs

    @classmethod
    def from_config(cls, config):
        return cls(**config)

   

"""
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
    
 """
