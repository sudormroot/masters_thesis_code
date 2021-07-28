import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import os
import sys 

libpath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "." 
sys.path.append(libpath)

import tensorflow as tf
from tensorflow import keras

#from tensorflow.keras.applications import ResNet50 as encoder_model
#from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications import VGG16 as encoder_model
from tensorflow.keras.applications.vgg16 import preprocess_input

#from tensorflow.keras.applications import MobileNet as encoder_model
#from tensorflow.keras.applications.MobileNet import preprocess_input

from imgaug_model import build_augmentation_model


class SupervisedModelDataGeneratorFromXy(keras.utils.Sequence):
    def __init__( self, 
                *,
                shuffle = True,
                batch_size = 32,
                output_shape = (128, 128, 3),
                max_iters = 1000000,
                subset = "train",
                X, y):


        print("data generator: output_shape = ", output_shape)

        input_shape = X[0].shape

        print("data generator: input_shape = ", input_shape)

        self.img_augmenter = build_augmentation_model(input_shape = input_shape,
                                                      output_shape = output_shape)

        self.batch_size = batch_size
        self.output_shape = output_shape
        
        self.data_IDs = list(range(len(y)))

        self.label_to_ID_dict = {yi: self.data_IDs[i] for i, yi in enumerate(y)}

        self.shuffle = shuffle
        
        self.X = X
        self.y = y

        self.n_classes = len(set(y))

        self.max_iters = max_iters

        self.subset = subset 

        self.on_epoch_end()

    def ID_to_label(self, ID):
        return self.label_to_ID_dict[ID]

    def on_epoch_end(self):

        self.indices = np.arange(len(self.data_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __len__(self):
        #return len(self.data_IDs) // self.batch_size
        #return 1000000 // self.batch_size
        if self.subset == "train":
            return self.max_iters * self.batch_size

        return 16 * self.batch_size

    def __getitem__(self, index):

        #index = 0
        #np.random.shuffle(self.indices)

        #batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_indices = np.random.choice(self.indices, self.batch_size, replace = True)

        # Find list of IDs
        batch_IDs = [self.data_IDs[k] for k in batch_indices]

        #print("index = ", index)
        #print("batch_indices = ", batch_indices)
        #print("batch_IDs = ", batch_IDs)


        # Generate data
        #X, y = self.__get_data(batch)
        X_batch = self.X[batch_IDs]

        X_batch = preprocess_input(X_batch)

        #y_batch = self.y[batch_IDs]
        y_batch = np.array(batch_IDs)

        return X_batch, y_batch


"""
    possible chopping_layer:
    conv2_block1_out
    conv2_block2_out
    conv2_block3_out

    conv3_block1_out
    conv3_block2_out
    conv3_block3_out
    conv3_block4_out

    conv4_block1_out
    conv4_block2_out
    conv4_block3_out
    conv4_block4_out
    conv4_block5_out
    conv4_block6_out

    conv5_block1_out
    conv5_block2_out
    conv5_block3_out
"""

def build_supervised_model( *, 
                            finetune = False, 
                            chopping_layer = None,  # indicating which layer we tap
                            learning_rate = 0.01,
                            embedding_input_shape, 
                            n_classes):

    """ embedding

    """
    inputs = keras.Input(shape = embedding_input_shape, name = "inputs")
    print("embedding_input_shape = ", embedding_input_shape)
    orig_embedding = encoder_model(
            include_top = False,      # no pre-defined head
            input_tensor = inputs,    # input tensor
            input_shape = embedding_input_shape, # input shape
            weights = "imagenet",
            pooling = "avg"           #max avg
            )

    #orig_embedding.summary()

    if chopping_layer is None:
        embedding_outputs = orig_embedding.layers[-1].output
    else:
        embedding_outputs = orig_embedding.get_layer(chopping_layer).output
        #embedding_outputs = keras.layers.GlobalMaxPooling2D()(embedding_outputs)
        embedding_outputs = keras.layers.GlobalAveragePooling2D()(embedding_outputs)

    model_embedding = keras.Model(inputs = inputs, outputs = embedding_outputs, name = "base-encoder")

    #for layer in model_embedding.layers:
    #    layer.trainable = False

    if finetune == True:
        model_embedding.trainable = True
    else:
        model_embedding.trainable = False

    """ newhead

    """
    
    DENSE_DIMS = (2 * n_classes, int(1.5 * n_classes))

    newhead_input_shape = model_embedding.layers[-1].output.shape[1:]

    print("newhead_input_shape = ", newhead_input_shape)
    print("DENSE_DIMS = ", DENSE_DIMS)

    inputs = keras.Input(shape = newhead_input_shape, name = "newhead_inputs")

    x = keras.layers.Flatten(name = "newhead_flatten")(inputs)

    for idx, dim in enumerate(DENSE_DIMS):
        #x = keras.layers.BatchNormalization()(x)   # We do not use BN since we found the
                                                    # performance improvement is limited.
        x = keras.layers.Dense(dim, activation = "relu", name = f"newhead_dense_{idx+1}")(x)
        x = keras.layers.Dropout(0.1, name = f"newhead_dropout_{idx+1}")(x)
        x = keras.layers.BatchNormalization()(x)


    #outputs = keras.layers.Dense(n_classes, activation = 'softmax', name = "newhead_output")(x)
    outputs = keras.layers.Dense(n_classes, activation = 'sigmoid', name = "newhead_output")(x)

    model_newhead = keras.Model(inputs = inputs, outputs = outputs, name = "newhead")

    #model_newhead.summary()


    """ complete model

    """

    inputs = keras.Input(shape = embedding_input_shape, name = "inputs")

    x = model_embedding(inputs)
    outputs = model_newhead(x)

    model = keras.Model(inputs = inputs, outputs = outputs)

    model.compile(  loss = 'sparse_categorical_crossentropy', 
                    optimizer = keras.optimizers.Adam(learning_rate = learning_rate), #'adam', 
                    metrics = ["accuracy"])

    #model.summary()

    return model

