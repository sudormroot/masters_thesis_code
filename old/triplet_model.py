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
import keras.backend as K

#from mbt_dataset import load_mbt_dataset_as_dataframe, load_mbt_dataset
#from mbt_split import split_mbt_dataset_as_dataframe, split_mbt_dataset
#from imgaug_model import build_augmentation_model

"""
def generate_triplets(*, 
                      batch_size = 32,
                      output_shape = (128, 128, 3),
                      img_augmenter,
                      X, y):
    
        
    X = X.astype('float32')
    
    #img_size = X[0].shape
    
    indices = list(range(len(X)))
    
    while True:
        #A = np.zeros((batch_size, *output_shape), dtype="float32")
        #P = np.zeros((batch_size, *output_shape), dtype="float32")
        #N = np.zeros((batch_size, *output_shape), dtype="float32")
        
        A = []
        P = []
        N = []
        
        for i in range(batch_size):
            
            idx = np.random.choice(indices, 2, replace = False)
                
            x1, x2 = X[idx]
            
            x1 = x1.copy()
            x2 = x2.copy()
            
            x1 = np.expand_dims(x1, axis = 0)
            x2 = np.expand_dims(x2, axis = 0)
            
            a = img_augmenter(x1)
            p = img_augmenter(x1)
            n = img_augmenter(x2)

            A.append(a[0])
            P.append(p[0])
            N.append(n[0])
            
            #A[i] = a[0]
            #P[i] = p[0]
            #N[i] = n[0]
            
        yield [np.array(A), np.array(P), np.array(N)]
        

def test_generate_triplets():
    #print(INPUT_SHAPE)

    X,y = load_mbt_dataset()
    #X = X.astype('float32')

    X_train, y_train, X_test, y_test = split_mbt_dataset(X, y, split = 0.5)

    img_augmenter = build_augmentation_model()

    for x in generate_triplets(img_augmenter = img_augmenter,
                           X = X_train,
                           y = y_train):
        a, p, n = x

        print()
        print(a.shape)
        print(p.shape)
        print(n.shape)

        print()
        print("|a-p| = ", tf.math.reduce_sum((a - p)**2).numpy())
        print("|a-n| = ", tf.math.reduce_sum((a - n)**2).numpy())

        break
"""

class TripletLossLayer(keras.layers.Layer):
    def __init__(self, alpha=0.2):
        super(TripletLossLayer, self).__init__()
        self.alpha = alpha

    def call(self, inputs):
        # a batch of (anchor, positive, negative), *after* encoding
        a, p, n = inputs

        #print("a.shape=", a.shape)
        #print("p.shape=", p.shape)
        #print("n.shape=", n.shape)
        
        #print("TripletLossLayer ", a.shape)
        
        # Euclidean distance in the embedding space. axis=-1 means sum
        # over the dimensions of the embedding, but don't sum over the
        # items of the batch. keepdims=True means the result eg d_ap
        # is of shape (batch_size, 1), not just (batch_size,).
        d_ap = K.sqrt(K.sum(K.square(a - p), axis=-1, keepdims=True))
        d_an = K.sqrt(K.sum(K.square(a - n), axis=-1, keepdims=True))

        # exactly as in the formula
        loss = K.maximum(0.0, d_ap - d_an + self.alpha)
        # loss is a tensor of shape (batch_size, 1), one loss per
        # triplet in the batch. This is the "expected" shape for Keras
        # losses, even though just a scalar, or just (batch_size,)
        # would also work.

        #print("loss=", loss)
        
        # this is the crucial line, allowing our calculation to be
        # used in the model
        self.add_loss(loss)
        # we won't use the return value, but let's return *something*
        return a

def test_TripletLossLayer():
    a = np.expand_dims(np.random.rand(5), axis = 0)
    p = np.expand_dims(np.random.rand(5), axis = 0)
    n = np.expand_dims(np.random.rand(5), axis = 0)


    TripletLossLayer(alpha = 0.5)([a, p, n])


# This code is from James McDermott's lecture in CT5133 Deep learning of NUIG 2020-2021
# I modified the original code a bit.

def build_triplet_model(*, 
                        #input_shape = (128, 128, 3),
                        alpha = 0.5, 
                        learning_rate = 0.001,
                        embedding_model):
    
    print()
    print("Building contrastive model with triplet loss ...")
    
    #inputs_shape = embedding_model.layers[0].output.shape.as_list()[1:]
    inputs_shape = (128, 128, 3)
    print("inputs_shape = ", inputs_shape)

    input_a = keras.Input(shape = inputs_shape)
    input_p = keras.Input(shape = inputs_shape)
    input_n = keras.Input(shape = inputs_shape)
    
    #print(input_a)
    
    
    # call the encoder three times to get embeddings
    a = embedding_model(input_a)
    p = embedding_model(input_p)
    n = embedding_model(input_n)

    # the return value from our TripletLossLayer is irrelevant
    # (NB return value is not the loss)
    dummy = TripletLossLayer(alpha = alpha)([a, p, n]) 
    model = keras.Model(inputs=[input_a, input_p, input_n], outputs=dummy)
    
    # compile with no loss, because TripletLossLayer has added the loss
    #model.compile(optimizer=keras.optimizers.Adam(learning_rate, decay = 1e-2 / 50), run_eagerly = False) 
    #model.compile(optimizer=keras.optimizers.Adam(learning_rate)) 
    model.compile(optimizer=keras.optimizers.Adam(learning_rate))  

    return model

